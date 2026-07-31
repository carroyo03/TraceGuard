"""Build TraceGuard as a regular wheel even for editable development installs.

Some macOS setups mark generated ``.pth`` files as hidden. Python then skips
them, making a conventional editable ``src`` installation unimportable.
"""

from collections.abc import Mapping

from flit_core import buildapi


def build_wheel(
    wheel_directory: str,
    config_settings: Mapping[str, object] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return buildapi.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: Mapping[str, object] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Produce a regular wheel to avoid fragile generated ``.pth`` files."""
    return build_wheel(wheel_directory, config_settings, metadata_directory)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: Mapping[str, object] | None = None,
) -> str:
    return buildapi.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: Mapping[str, object] | None = None,
) -> str:
    return prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def get_requires_for_build_wheel(
    config_settings: Mapping[str, object] | None = None,
) -> list[str]:
    return buildapi.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(
    config_settings: Mapping[str, object] | None = None,
) -> list[str]:
    return get_requires_for_build_wheel(config_settings)
