"""Skill management for the Extended OpenAI Conversation component."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any

import yaml

from homeassistant.core import HomeAssistant

from .const import DEFAULT_SKILLS_DIRECTORY, SKILL_FILE_NAME

_LOGGER = logging.getLogger(__name__)


@dataclass
class Skill:
    """Represents a skill loaded from SKILL.md.

    Only metadata (name, description) is loaded initially.
    The full content (body) is loaded on-demand via read_skill function.
    """

    name: str
    description: str
    directory: Path | None = None

    def __post_init__(self):
        """Validate skill name."""
        if not self.name:
            raise ValueError("Skill name is required")
        if len(self.name) > 64:
            raise ValueError("Skill name must be 64 characters or less")
        if len(self.description) > 1024:
            raise ValueError("Skill description must be 1024 characters or less")


class SkillMdParser:
    """Parser for SKILL.md files following Agent Skills standard."""

    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    @classmethod
    def parse(cls, content: str, directory: Path) -> Skill | None:
        """Parse SKILL.md content and return a Skill object.

        Only parses frontmatter (name, description).
        The body content is loaded on-demand via read_skill function.

        Args:
            content: Full content of SKILL.md file
            directory: Path to the skill directory

        Returns:
            Skill object with metadata, or None if parsing fails
        """
        match = cls.FRONTMATTER_PATTERN.match(content)
        if not match:
            _LOGGER.warning(
                "Invalid SKILL.md format in %s: missing frontmatter", directory
            )
            return None

        try:
            frontmatter = yaml.safe_load(match.group(1))
        except yaml.YAMLError as e:
            _LOGGER.warning("Failed to parse YAML frontmatter in %s: %s", directory, e)
            return None

        if not isinstance(frontmatter, dict):
            _LOGGER.warning("Invalid frontmatter format in %s", directory)
            return None

        name = frontmatter.get("name")
        description = frontmatter.get("description")

        if not name or not description:
            _LOGGER.warning(
                "Missing required fields (name, description) in %s", directory
            )
            return None

        try:
            return Skill(
                name=name,
                description=description,
                directory=directory,
            )
        except ValueError as e:
            _LOGGER.warning("Invalid skill in %s: %s", directory, e)
            return None

    @classmethod
    def extract_body(cls, content: str) -> str:
        """Extract the body content after frontmatter.

        Args:
            content: Full content of SKILL.md file

        Returns:
            Body content (markdown after frontmatter)
        """
        match = cls.FRONTMATTER_PATTERN.match(content)
        if not match:
            return content

        return content[match.end() :].strip()


class SkillManager:
    """Manages skills for the Extended OpenAI Conversation component.

    Skills are loaded from the skills/ directory.
    Only metadata (name, description) is loaded initially for system prompt.
    Full skill content is loaded on-demand via read_skill function.

    This class is designed to be used as a singleton - it only handles
    skill loading and reading. Enabled skills list is managed
    per ConversationEntity via subentry.data[CONF_SKILLS].
    """

    _instance: SkillManager | None = None

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the skill manager.

        Args:
            hass: Home Assistant instance
        """
        self._hass = hass
        self._skills: dict[str, Skill] = {}
        self._skills_dir: Path | None = None

    @classmethod
    async def async_get_instance(cls, hass: HomeAssistant) -> SkillManager:
        """Get or create the singleton instance and load skills.

        Args:
            hass: Home Assistant instance

        Returns:
            The singleton SkillManager instance with skills loaded
        """
        if cls._instance is None:
            cls._instance = cls(hass)
            await cls._instance.async_load_skills()
        return cls._instance

    @property
    def skills_dir(self) -> Path:
        """Get the skills directory path."""
        if self._skills_dir is None:
            self._skills_dir = (
                Path(self._hass.config.config_dir)
                / "custom_components"
                / "extended_openai_conversation"
                / DEFAULT_SKILLS_DIRECTORY
            )
        return self._skills_dir

    async def async_load_skills(self) -> None:
        """Load all skills from the skills directory.

        Only loads metadata (frontmatter) from SKILL.md files.
        """
        self._skills.clear()

        # Run all blocking file system operations in executor
        skills_data = await self._hass.async_add_executor_job(self._load_skills_sync)

        for skill_dir, content in skills_data:
            try:
                skill = SkillMdParser.parse(content, skill_dir)
                if skill:
                    self._skills[skill.name] = skill
                    _LOGGER.debug("Loaded skill: %s from %s", skill.name, skill_dir)
            except Exception as e:
                _LOGGER.exception(
                    "Unexpected error loading skill from %s: %s", skill_dir, e
                )

        _LOGGER.info("Loaded %d skills from %s", len(self._skills), self.skills_dir)

    def _load_skills_sync(self) -> list[tuple[Path, str]]:
        """Load skill files synchronously (run in executor).

        Returns:
            List of tuples (skill_directory, file_content)
        """
        results: list[tuple[Path, str]] = []

        if not self.skills_dir.exists():
            _LOGGER.debug("Skills directory does not exist: %s", self.skills_dir)
            return results

        if not self.skills_dir.is_dir():
            _LOGGER.warning("Skills path is not a directory: %s", self.skills_dir)
            return results

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / SKILL_FILE_NAME
            if not skill_file.exists():
                _LOGGER.debug("No SKILL.md found in %s", skill_dir)
                continue

            try:
                content = skill_file.read_text(encoding="utf-8")
                results.append((skill_dir, content))
            except OSError as e:
                _LOGGER.warning("Failed to read skill file %s: %s", skill_file, e)

        return results

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name.

        Args:
            name: The skill name

        Returns:
            The skill if found, None otherwise
        """
        return self._skills.get(name)

    def get_all_skills(self) -> list[Skill]:
        """Get all loaded skills.

        Returns:
            List of all skills
        """
        return list(self._skills.values())

    def build_skills_prompt_section(
        self, enabled_skill_names: list[str] | None = None
    ) -> str:
        """Build the skills section for the system prompt in XML format.

        Only includes skill metadata (name, description) for enabled skills.
        This is Level 1 of progressive disclosure.

        Args:
            enabled_skill_names: List of enabled skill names.
                                If None or empty, all skills are enabled.

        Returns:
            Skills section to append to system prompt in XML format
        """
        if enabled_skill_names is None:
            enabled_skill_names = []

        # If no skills specified, enable all; otherwise filter by list
        if enabled_skill_names:
            enabled_skills = [
                skill
                for skill in self._skills.values()
                if skill.name in enabled_skill_names
            ]
        else:
            enabled_skills = list(self._skills.values())

        if not enabled_skills:
            return ""

        lines = [
            "",
            "# Available Skills",
            "",
            "Use the read_skill function to get detailed instructions when needed:",
            "",
        ]
        lines.append("<skills>")
        for skill in enabled_skills:
            lines.append(f'  <skill name="{skill.name}">')
            lines.append(f"    <description>{skill.description}</description>")
            lines.append("  </skill>")
        lines.append("</skills>")
        lines.append("</available_skills>")

        return "\n".join(lines)

    def get_skill_functions(
        self, enabled_skill_names: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Get the skill-related function definitions.

        Args:
            enabled_skill_names: List of enabled skill names.
                                If None or empty, all skills are enabled.

        Returns:
            List of function specs for read_skill and execute_skill_script
        """
        if enabled_skill_names is None:
            enabled_skill_names = []

        # If no skills specified, enable all; otherwise filter by list
        if enabled_skill_names:
            enabled_skills = [
                skill
                for skill in self._skills.values()
                if skill.name in enabled_skill_names
            ]
        else:
            enabled_skills = list(self._skills.values())
        if not enabled_skills:
            return []

        return [
            {
                "spec": {
                    "name": "read_skill",
                    "description": (
                        "Read skill content. Without file_path, reads SKILL.md instructions. "
                        "With file_path, reads a specific file from the skill directory."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "description": "Name of the skill",
                            },
                            "file_path": {
                                "type": "string",
                                "description": (
                                    "Optional. Relative path to file (e.g., reference.md). "
                                    "If omitted, reads SKILL.md body."
                                ),
                            },
                        },
                        "required": ["skill_name"],
                    },
                },
                "function": {
                    "type": "skill_read",
                    "skills_dir": str(self.skills_dir),
                },
            },
            {
                "spec": {
                    "name": "execute_skill_script",
                    "description": (
                        "Execute a shell command or script provided by a skill. "
                        "Commands run in the skill's directory context."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "description": "Name of the skill",
                            },
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute (relative to skill directory)",
                            },
                        },
                        "required": ["skill_name", "command"],
                    },
                },
                "function": {
                    "type": "skill_exec",
                    "skills_dir": str(self.skills_dir),
                },
            },
        ]
