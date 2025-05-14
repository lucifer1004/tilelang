import ctypes
import os
import sys
from typing import List, Literal, Optional, Tuple, Union

from tvm.target import Target

from .nvcc import get_target_compute_version


def _get_nvrtc_version(cuda_version: int) -> str:
    # TODO: Expose this from native code
    # Follows same logic as LazyNVRTC.cpp getLibVersion()
    major = cuda_version // 1000
    minor = (cuda_version // 10) % 10

    if sys.platform == "win32":
        if major < 11 or (major == 11 and minor < 3):
            return f"{major}{minor}"
        elif major == 11:
            return "112"
        else:
            return f"{major}0"
    else:
        if major < 11 or (major == 11 and minor < 3):
            return f"{major}.{minor}"
        elif major == 11:
            return "11.2"
        else:
            return str(major)

# Load CUDA driver and NVRTC
def _get_cuda_library() -> ctypes.CDLL:
    if sys.platform == "win32":
        return ctypes.CDLL("nvcuda.dll")
    else:  # Unix-based systems
        return ctypes.CDLL("libcuda.so.1")
    
def _get_nvrtc_library() -> ctypes.CDLL:
    # Get NVRTC version based on CUDA runtime version
    # Use an alternative approach to get the CUDA version
    # since cudart().getVersion() is failing
    import torch

    try:
        import torch.cuda

        cuda_runtime_version = torch.cuda.cudart().getVersion()
    except (ImportError, AttributeError):
        # Fallback: if we have CUDA available, get version from device properties
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            # Import locally to avoid circular imports
            import torch.cuda

            props = torch.cuda.get_device_properties(
                torch.cuda.current_device())
            cuda_runtime_version = props.major * 1000 + props.minor * 10
        else:
            # Hardcode a default CUDA version if all else fails
            cuda_runtime_version = 12000  # Assume CUDA 12.0 as default

    version = _get_nvrtc_version(cuda_runtime_version)

    if sys.platform == "win32":
        # Windows handling remains the same
        lib_name = f"nvrtc64_{version}_0.dll"
        return ctypes.CDLL(lib_name)
    else:
        lib_paths = [
            f"libnvrtc.so.{version}",
            os.path.join(
                os.environ.get("CUDA_HOME", ""), f"lib64/libnvrtc.so.{version}"
            ),
            "/usr/local/cuda/lib64/libnvrtc.so",
        ]

        for path in lib_paths:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue

        raise RuntimeError(
            "Could not find libnvrtc.so. Please make sure CUDA is installed."
        )

def get_nvrtc_version() -> Tuple[int, int]:
    result, major, minor = nvrtc.nvrtcVersion()
    assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get NVRTC version: {result}"
    return (major, minor)

NVRTC_SUCCESS = 0

class NVRTCCompiler(object):
    _instance = None
    
    @staticmethod
    def check_nvrtc(result: int) -> None:
        if result != NVRTC_SUCCESS:
            err_str = ctypes.c_char_p()
            libnvrtc.nvrtcGetErrorString(result, ctypes.byref(err_str))
            error_message = (
                err_str.value.decode()
                if err_str.value is not None
                else "Unknown CUDA error"
            )
            raise RuntimeError(f"CUDA error: {error_message}")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NVRTCCompiler, cls).__new__(cls)
            cls._instance.cuda = _get_cuda_library()
            cls._instance.nvrtc = _get_nvrtc_library()
        return cls._instance

    def compile(self, code: str,
                 target_format: Literal["ptx", "cubin"] = "ptx",
                 arch: Optional[int] = None,
                 options: Optional[Union[str, List[str]]] = None,
                 verbose: bool = False) -> bytearray:
        """Compile cuda code with NVRTC.

        Parameters
        ----------
        code : str
            The cuda code.

        target_format : Literal["ptx", "cubin"]
            The target format of nvrtc compiler.

        arch : Optional[int]
            The cuda architecture code.

        options : Optional[Union[str, List[str]]]
            The additional options.

        verbose : bool
            Whether to print the verbose output.

        Return
        ------
        result_bytes : bytearray
            The bytearray of the cubin or ptx code.
        """
        if arch is None:
            # If None, then it will use `tvm.target.Target.current().arch`.
            # Target arch could be a str like "80", "90", "90a", etc.
            compute_version = "".join(
                get_target_compute_version(Target.current(allow_none=True)).split("."))
            arch = int(compute_version)
        prefix = "compute" if target_format == "ptx" else "sm"
        suffix = "a" if arch >= 90 else ""
        arch_option = f"--gpu-architecture={prefix}_{arch}{suffix}"

        file_name = "tvm_kernels"
        if target_format not in ["cubin", "ptx"]:
            raise ValueError("target_format must be cubin or ptx")

        final_options = ["-default-device"]
        if get_nvrtc_version() >= (12, 8):
            final_options += ["-pch"]
        if arch is not None:
            final_options += [arch_option]

        if options:
            if isinstance(options, str):
                final_options += [options]
            elif isinstance(options, list):
                final_options += options
            else:
                raise ValueError("options must be str or list of str")

        code = "#include <tl_templates/cuda/nvrtc_std.h>\n" + code
        code_bytes = bytes(code, "utf-8")
        program = ctypes.c_void_p()
        self.check_nvrtc(
            self.nvrtc.nvrtcCreateProgram(
                ctypes.byref(program),
                code_bytes,
                bytes(file_name, "utf-8"),
                0,
                None,
                None,
            )
        )

        options_bytes = [bytes(flag, "utf-8") for flag in final_options]
        options_carray = (ctypes.c_char_p * len(options_bytes))(*options_bytes)
        compile_result = self.nvrtc.nvrtcCompileProgram(program, len(options_bytes), options_carray)

        if compile_result != NVRTC_SUCCESS:
            msg = f"{code}\n" \
                f"Compilation error:\n"
            if verbose:
                log_size = ctypes.c_size_t()
                result = self.nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
                assert result == NVRTC_SUCCESS, f"Failed to get program log size: {result}"
                log_bytes = ctypes.create_string_buffer(log_size.value)
                result = self.nvrtc.nvrtcGetProgramLog(program, log_bytes)
                assert result == NVRTC_SUCCESS, f"Failed to get program log: {result}"
                msg += f"{log_bytes.decode('utf-8')}\n"
            else:
                msg += "Turn on verbose to see the full compilation log."
            msg += f"Options: {' '.join(final_options)}\n"
            raise RuntimeError(msg)

        if target_format == "cubin":
            cubin_size = ctypes.c_size_t()
            result = self.nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(cubin_size))
            assert result == NVRTC_SUCCESS, f"Failed to get CUBIN size: {result}"
            result_bytes = ctypes.create_string_buffer(cubin_size.value)
            result = self.nvrtc.nvrtcGetCUBIN(program, result_bytes)
            assert result == NVRTC_SUCCESS, f"Failed to get CUBIN: {result}"
        else:
            ptx_size = ctypes.c_size_t()
            result = self.nvrtc.nvrtcGetPTXSize(program, ctypes.byref(ptx_size))
            assert result == NVRTC_SUCCESS, f"Failed to get PTX size: {result}"
            result_bytes = ctypes.create_string_buffer(ptx_size.value)
            result = self.nvrtc.nvrtcGetPTX(program, result_bytes)
            assert result == NVRTC_SUCCESS, f"Failed to get PTX: {result}"

        # Destroy handler
        assert self.nvrtc.nvrtcDestroyProgram(ctypes.byref(program)) == NVRTC_SUCCESS, f"Failed to destroy program: {result}"

        return result_bytes
