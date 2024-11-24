# Standard Library
import argparse
import codecs, json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
from PIL import Image

# Third Party
import torch

# CosyPose
from happypose.pose_estimators.cosypose.cosypose.integrated.pose_estimator import (
    PoseEstimator as CosyPoseEstimator,
)

from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import (
    CosyPoseWrapper,
)

# MegaPose
from happypose.pose_estimators.megapose.inference.pose_estimator import (
    PoseEstimator as MegaPoseEstimator,
)

# HappyPose
from happypose.toolbox.inference.types import DetectionsType, ObservationTensor
from happypose.toolbox.inference.utils import filter_detections, load_detector
from happypose.toolbox.utils.logging import get_logger, set_logging_level
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.inference.types import DetectionsType, PoseEstimatesType
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.inference.utils import make_detections_from_object_data
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model
from happypose.toolbox.inference.example_inference_utils import (
    load_object_data,
    make_detections_visualization,
    make_poses_visualization,    
    save_predictions,
)

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_object_dataset(
    mesh_dir: Path, mesh_units= str
) -> RigidObjectDataset:
    """
    TODO
    """

    rigid_objects = []
    #mesh_units = "m" or "mm"
    assert mesh_dir.exists(), f"Missing mesh directory {mesh_dir}"

    for mesh_path in mesh_dir.iterdir():
        if mesh_path.suffix in {".obj", ".ply"}:
            obj_name = mesh_path.with_suffix("").name
            rigid_objects.append(
                RigidObject(label=obj_name, mesh_path=mesh_path, mesh_units=mesh_units),
            )
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def setup_cosypose_estimator(dataset_to_use: str, object_dataset: RigidObjectDataset):
    # TODO: remove this wrapper from code base
    cosypose = CosyPoseWrapper(
        dataset_name=dataset_to_use, model_type="synth+real", object_dataset=object_dataset, n_workers=8
    )   #model_type=pbr|synth+real

    return cosypose.detector, cosypose.pose_predictor


def setup_megapose_estimator(model_name: str, object_dataset: RigidObjectDataset):
    logger.info(f"Loading model {model_name}.")
    model_info = NAMED_MODELS[model_name]
    pose_estimator = load_named_model(model_name, object_dataset).to(device)
    # Speed up things by subsampling coarse grid
    pose_estimator._SO3_grid = pose_estimator._SO3_grid[::8]

    return pose_estimator, model_info


def run_cosypose_inference(
    pose_estimator: CosyPoseEstimator,
    observation: ObservationTensor,
    detections: DetectionsType,
) -> None:
    observation.to(device)
    if not detections.infos.empty:      #Mod to exit when no object is detected
        data_TCO, extra_data = pose_estimator.run_inference_pipeline(
            observation=observation, detections=detections, n_refiner_iterations=3
        )
        logger.info("Timings:")
        logger.info(extra_data["timing_str"])

        return data_TCO.cpu()
    else:
        logger.warning("No detections found, returning None")
        return None

def run_megapose_inference(
    pose_estimator: MegaPoseEstimator,
    model_info: Dict,
    observation: ObservationTensor,
    detections: DetectionsType,
) -> None:
    observation.to(device)
    if not detections.infos.empty:      #Mod to exit when no object is detected
        data_TCO, extra_data = pose_estimator.run_inference_pipeline(
            observation=observation, detections=detections, **model_info["inference_parameters"],
        )
        logger.info("Timings:")
        logger.info(extra_data["timing_str"])

        return data_TCO.cpu()
    else:
        logger.warning("No detections found, returning None")
        return None


def output_pose(
    pose_estimates: PoseEstimatesType,
    main_dir: Path,
    output_file: str,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    
    # Take just first occurrence
    t = Transform(poses[0])
    q = np.array([t.quaternion.x, t.quaternion.y, t.quaternion.z, t.quaternion.w])
    a = np.concatenate((q, t.translation), axis=None)
    b = a.tolist() # nested lists with same data, indices
    output_fn = main_dir / output_file
    output_fn.parent.mkdir(exist_ok=True)
    json.dump(b, codecs.open(output_fn, 'w', encoding='utf-8'), 
        separators=(',', ':'), 
        sort_keys=True, 
        indent=4) ### this saves the array in .json format

    logger.info(f"Wrote predictions: {output_fn}")


def load_camera_image(
    main_dir: Path,
    load_depth: bool = False,
    camera_data_name: str = "camera_data.json",
    rgb_name: str = "image_rgb.png",
    depth_name: str = "image_depth.png",
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((main_dir / camera_data_name).read_text())

    rgb = np.array(Image.open(main_dir / rgb_name), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(main_dir / depth_name), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data    

def load_bb_data(data_path: Path) -> List[ObjectData]:
    """"""
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data

def load_boundingbox(
    main_dir: Path,
    object_name: str = "object",
) -> DetectionsType:

    input_object_data = load_bb_data(main_dir / "boundingbox.json")
    import re
    input_object_data[0].label = re.sub('[(,\')]', '', object_name)
    #input_object_data[0].label = object_name,  #override object name in json file, it's name of PLY or OBJ file
    print(input_object_data[0].label)
    detections = make_detections_from_object_data(input_object_data)
    return detections


def save_predictions(
    pose_estimates: PoseEstimatesType,
    example_dir: Path,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose))
        for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = example_dir / "visualizations/object_data_inf.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")

def main():
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("object_name")
    parser.add_argument("objects_dataset_folder")
    parser.add_argument("output_file")
    parser.add_argument("--dataset", type=str, default="ycbv")
    parser.add_argument("--detection-type", type=str, default="cosypose")   #Use cosypose or megapose inference
    parser.add_argument("--mesh-unit", type=str, default="m")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--vis-poses", action="store_true")
    args = parser.parse_args()

    main_dir = Path(args.objects_dataset_folder)
    object_data_dir = main_dir / "Dataset" / args.object_name
    assert (
        object_data_dir.exists()
    ), "Example {args.object_name} not available, follow download instructions"
    # dataset_to_use = args.dataset  # hope/tless/ycbv

    # Load data
    detections = load_boundingbox(main_dir, args.object_name).to(device)
    object_dataset = make_object_dataset(object_data_dir, args.mesh_unit)
    rgb, depth, camera_data = load_camera_image(main_dir, load_depth=False)
    # TODO: cosypose forward does not work if depth is loaded detection
    # contrary to megapose
    observation = ObservationTensor.from_numpy(rgb, depth=None, K=camera_data.K).to(
        device
    )

    # Load models
    logger.info(f"Running inference for object \"{args.object_name.upper()}\" with {args.detection_type.upper()}")

    output = None
    if (args.detection_type == "cosypose"): 
        
        # Use CosyPose inference        
        detector, pose_estimator = setup_cosypose_estimator("ycbv", object_dataset) #args.dataset, use always ycbv for cosypose

        if (args.dataset == "ycbv"):
            detections = detector.get_detections(observation, output_masks=True)
            available_labels = [obj.label for obj in object_dataset.list_objects]
            detections = filter_detections(detections, available_labels)
        
        output = run_cosypose_inference(pose_estimator, observation, detections)
        
    else:       
        
        # Use MegaPose inference
        pose_estimator, model_info = setup_megapose_estimator("megapose-1.0-RGB", object_dataset)

        if (args.dataset == "ycbv"):
            # TODO: hardcoded detector
            detector = load_detector(run_id="detector-bop-ycbv-pbr--970850", device=device)

            detections = detector.get_detections(observation, output_masks=True)
            available_labels = [obj.label for obj in object_dataset.list_objects]
            detections = filter_detections(detections, available_labels)
        
        output = run_megapose_inference(pose_estimator, model_info, observation, detections)

    # Output predictions
    if output is not None:
        output_pose(output, main_dir, args.output_file)        
    else:
        logger.warning("No predictions found, skipping output")
        exit(1)

    if args.vis_detections:
        make_detections_visualization(rgb, detections, main_dir)

    if args.vis_poses:
        save_predictions(output, main_dir)
        object_datas = load_object_data(main_dir / "visualizations/object_data_inf.json")
        make_poses_visualization(
            rgb, object_dataset, object_datas, camera_data, main_dir
        )


if __name__ == "__main__":
    main()
