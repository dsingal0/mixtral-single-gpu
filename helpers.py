import os
import io
import re
import base64
import uuid
import sys
import logging
import codecs
import numpy as np
from PIL import Image

DEFAULT_MAX_NVCF_MSG_SIZE = 5 * 1000 * 1000  # 5MB
IMAGE_FORMAT = "JPEG"
IMAGE_QUALITY = 90

b64_pattern = re.compile("^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)?$")


def create_transaction_id():
    return str(uuid.uuid4())


def get_logger():
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [INFERENCE] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    return logger


def get_scalar_inputs(request, pairs: list):
    """
    converts tensor request object into numpy values or default values
    :param request: Triton request Tensor object
    :param pairs: list of tuple key/default value or single key
    :return:
    """
    import triton_python_backend_utils as pb_utils  # only available inside Triton container

    inputs = []
    for p in pairs:
        # split pair into key and value
        if len(p) == 2:
            k, v = p
        elif len(p) == 1:
            # if only one value is given, treat as a key
            k, v = p[0], None
        else:
            raise ValueError(f"Incorrect pair (f{p}) was passed to get input")

        input = pb_utils.get_input_tensor_by_name(request, k)
        if input is None:
            # set value as default
            input = v
        else:
            # get numpy value
            input = input.as_numpy().item()

        # handle byte input
        if isinstance(input, bytes):
            input = codecs.decode(input, "utf-8", "ignore")

        inputs.append(input)
    return inputs


def get_output_path(request_parameters: dict):
    request_id = get_request_id(request_parameters)
    output_path = request_parameters.get(
        "NVCF-LARGE-OUTPUT-DIR", f"/var/inf/response/{request_id}"
    )
    return output_path


def get_input_path(request_parameters: dict):
    request_id = get_request_id(request_parameters)
    input_assets_path_base = request_parameters.get(
        "NVCF-ASSET-DIR", f"/var/inf/inputAssets/{request_id}"
    )
    return input_assets_path_base


def get_max_msg_size(request_parameters: dict):
    max_response_size = request_parameters.get("NVCF-MAX-RESPONSE-SIZE-BYTES")
    if max_response_size is None:
        l = get_logger()
        l.warning(
            "Could not find 'NVCF-MAX-RESPONSE-SIZE-BYTES' in request parameters "
            f"defaulting to {DEFAULT_MAX_NVCF_MSG_SIZE}!"
        )
        max_response_size = DEFAULT_MAX_NVCF_MSG_SIZE

    return float(max_response_size)


def get_nca_id(request_parameters: dict):
    return request_parameters.get("NVCF-NCAID", "")


def get_request_id(request_parameters: dict):
    return request_parameters.get("NVCF-REQID", "")


def get_asset_id(request_parameters: dict):
    return request_parameters.get("NVCF-FUNCTION-ASSET-IDS", "")


def get_nspect_id(request_parameters: dict):
    return request_parameters.get("NVCF-NSPECTID", "")


def get_properties_sub(request_parameters: dict):
    return request_parameters.get("NVCF-SUB", "")


def get_instance_type(request_parameters: dict):
    return request_parameters.get("NVCF-INSTANCETYPE", "")


def get_backend(request_parameters: dict):
    return request_parameters.get("NVCF-BACKEND", "")


def get_function_id(request_parameters: dict):
    return request_parameters.get("NVCF-FUNCTION-ID", "")


def get_function_version_id(request_parameters: dict):
    return request_parameters.get("NVCF-FUNCTION-VERSION-ID", "")


def get_env(request_parameters: dict):
    return request_parameters.get("NVCF-ENV", "")


def get_config_value(value_name: str, model_config: dict = None):
    """
    returns a value from Triton's model config or from environment variable with the priority given to the environment
    """
    if model_config is None:
        return os.environ[value_name]
    else:
        return os.environ.get(
            value_name, model_config["parameters"][value_name]["string_value"]
        )


def load_npz(input_str: str, root_dir: str, array_name: str):
    """
    Loads an nzp from a path
    :param input_str: path to npz
    :param root_dir: directory where asset are saved
    :return: a numpy array
    """
    if os.path.exists(os.path.join(root_dir, input_str)):
        try:
            data = np.load(os.path.join(root_dir, input_str))
            return data[array_name]
        except Exception as e:
            raise Exception(f"{input_str} was not a file path of an npz file. {e}")
    else:
        raise Exception(f"Unsure what {input_str} is!")


def load_image(input_str: str, root_dir: str, has_transparency: bool = False):
    """
    Loads an image from a b64 string or from a path
    :param input_str: b64 string or path to image
    :param root_dir: directory where images are saved
    :param has_transparency: if the alpha channel should be kept.
    :return: a PIL Image
    """
    if os.path.exists(os.path.join(root_dir, input_str)):
        # image exists in path
        try:
            i = Image.open(os.path.join(root_dir, input_str))
        except Exception as e:
            raise Exception(f"{input_str} was not a file path of an image. {e}")
    elif b64_pattern.match(input_str):
        # image might be a b64 string
        try:
            i = Image.open(io.BytesIO(base64.decodebytes(bytes(input_str, "utf-8"))))
        except Exception as e:
            raise Exception(f"{input_str} was not a b64 encoded image string. {e}")
    else:
        raise Exception(f"Unsure what {input_str} is!")

    if has_transparency:
        return i.convert("RGBA")
    else:
        return i.convert("RGB")


def encode_bytes_base64_to_str(b: bytes):
    return base64.b64encode(b)


def encode_image_to_base64(
    image: Image, image_format: str = "JPEG", image_quality: int = IMAGE_QUALITY
):
    """
    accepts an PIL Image and returns a base64 encoded representation of image
    """
    raw_bytes = io.BytesIO()
    if image_format == "JPEG":
        image.save(raw_bytes, image_format, quality=image_quality, optimize=True)
    elif image_format == "PNG":
        image.save(raw_bytes, image_format, optimize=True)
    else:
        image.save(raw_bytes, image_format)
    raw_bytes.seek(0)  # return to the start of the buffer
    return encode_bytes_base64_to_str(raw_bytes.read())


def decode_base64_str_to_bytes(base64_str: str):
    return io.BytesIO(base64.decodebytes(base64_str))


def decode_base64_to_image(base64_str: str, has_transparency: bool = False):
    """
    accepts base64 encoded representation of image and returns PIL Image
    :param base64_str: a string of image file bytes encoded in base64
    :param has_transparency: if the alpha channel should be kept.
    :return: a PIL Image
    """
    i = Image.open(decode_base64_str_to_bytes(base64_str))
    if has_transparency:
        return i.convert("RGBA")
    else:
        return i.convert("RGB")


def save_image_with_directory(
    image: Image,
    path: str = "",
    image_format: str = "JPEG",
    image_quality: int = IMAGE_QUALITY,
):
    """
    Saves an image at the specified path, creating any directories that do not exist.
    """
    # Make sure the directory and parent directories exist
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if image_format == "JPEG":
        extension = "jpg"
    elif image_format == "PNG":
        extension = "png"
    else:
        raise ValueError(
            "Unexpected image format {}. Currently only JPEG or PNG is supported!"
        )

    save_path = os.path.join(path, f"image.{extension}")
    if image_format == "JPEG":
        image.save(save_path, image_format, quality=image_quality, optimize=True)
    else:
        image.save(save_path, image_format)
    return save_path
