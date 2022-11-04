import os, hashlib, pkg_resources, subprocess
import torch

from .generic import get_unique_timestamp

# these are the base keys for the model dictionary to save
model_dict_full = {
    "epoch": 0,
    "model_state_dict": None,
    "optimizer_state_dict": None,
    "loss": None,
    "timestamp": None,
    "timestamp_hash": None,
    "git_hash": None,
    "version": None,
}

model_dict_required = {
    "model_state_dict": None,
    "optimizer_state_dict": None,
}

best_model_path_end = "_best.pth.tar"


def save_model(model_dict, model, params, path, onnx_export=True):
    """
    Save the model dictionary to a file.

    Args:
        model_dict (dict): Model dictionary to save.
        model (torch model): Trained torch model.
        params (dict): The parameter dictionary.
        path (str): The path to save the model dictionary to.
        onnx_export (bool): Whether to export to ONNX and OpenVINO.
    """
    num_channel = params["model"]["num_channels"]
    model_dimension = params["model"]["dimension"]
    ov_output_data_type = params["model"].get("data_type", "FP32")
    input_shape = params["patch_size"]

    model_dict["timestamp"] = get_unique_timestamp()
    model_dict["timestamp_hash"] = hashlib.sha256(
        str(model_dict["timestamp"]).encode("utf-8")
    ).hexdigest()
    model_dict["version"] = pkg_resources.require("GANDLF")[0].version
    try:
        model_dict["git_hash"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        model_dict["git_hash"] = None
    torch.save(model_dict, path)

    if not (onnx_export):
        if "onnx_print" not in params:
            print("WARNING: Current model is not supported by ONNX/OpenVINO!")
            params["onnx_print"] = True
        return
    else:
        try:
            onnx_path = path.rstrip("pth.tar") + "onnx"
            if model_dimension == 2:
                dummy_input = torch.randn(
                    (1, num_channel, input_shape[0], input_shape[1])
                )
            else:
                dummy_input = torch.randn(
                    (1, num_channel, input_shape[0], input_shape[1], input_shape[2])
                )

            with torch.no_grad():
                torch.onnx.export(
                    model.to("cpu"),
                    dummy_input.to("cpu"),
                    onnx_path,
                    opset_version=11,
                    export_params=True,
                    verbose=True,
                    input_names=["input"],
                    output_names=["output"],
                )
        except RuntimeWarning:
            print("WARNING: Cannot export to ONNX model.")
            return

        try:
            from openvino.runtime import Core, serialize
            core = Core()
            model = core.read_model(onnx_path)

            # There is an alternative way to trigger FP16 execution on GPU in openvino/master
            # In future avoid using this code under the next if-block for model export, export it as-is
            if ov_output_data_type.upper() == 'FP16':
                from openvino.offline_transformations import compress_model_transformation
                compress_model_transformation(model)

            ov_path = path.rstrip('onnx') + 'xml'
            serialize(model, ov_path)
        except:
            print("WARNING: Conversion to OpenVINO IR failed.")


def load_model(path, device, full_sanity_check=True):
    """
    Load a model dictionary from a file.

    Args:
        path (str): The path to save the model dictionary to.
        device (torch.device): The device to run the model on.
        full_sanity_check (bool): Whether to run full sanity checking on model.

    Returns:
        dict: Model dictionary containing model parameters and metadata.
    """
    model_dict = torch.load(path, map_location=device)

    # check if the model dictionary is complete
    if full_sanity_check:
        incomplete_keys = [
            key for key in model_dict_full.keys() if key not in model_dict.keys()
        ]
        if len(incomplete_keys) > 0:
            raise RuntimeWarning(
                "Model dictionary is incomplete; the following keys are missing:",
                incomplete_keys,
            )

    # check if required keys are absent, and if so raise an error
    incomplete_required_keys = [
        key for key in model_dict_required.keys() if key not in model_dict.keys()
    ]
    if len(incomplete_required_keys) > 0:
        raise KeyError(
            "Model dictionary is incomplete; the following keys are missing:",
            incomplete_required_keys,
        )

    return model_dict


def load_ov_model(path, device="CPU"):
    """
    Load an OpenVINO IR model from an .xml file.

    Args:
        path (str): The path to the OpenVINO .xml file.
        device (str): The device to run inference, can be "CPU", "GPU" or "MULTI:CPU,GPU". Default to be "CPU".

    Returns:
        exec_net (OpenVINO executable net): executable OpenVINO model.
        input_blob (str): Input name.
        output_blob (str): Output name.
    """

    try:
        from openvino import runtime as ov
    except ImportError:
        raise ImportError("OpenVINO inference engine is not configured correctly.")

    core = ov.Core()
    if device.lower() == "cuda":
        device = "GPU"

    if device == "GPU":
        core.set_property({"CACHE_DIR": os.path.dirname(os.path.abspath(path))})
        # core.set_property("GPU", ov.hint.inference_precision(ov.Type.f16))  # available in openvino/master

    model = core.read_model(path)
    compiled_model = core.compile_model(model, device.upper())
    input_layer = compiled_model.inputs
    output_layer = compiled_model.outputs

    return compiled_model, input_layer, output_layer
