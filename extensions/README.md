# Falcon Extensions

This directory contains custom extensions for the Falcon pipeline. Each subdirectory provides custom processors and/or data types.

To create a new extension, the `template_extension` directory can be duplicated and modified.

By default, contents within this folder are excluded from the main Falcon Git repository via `.gitignore`. This architecture allows individual extension folders to be managed as independent Git repositories. Code can be saved, tracked, and shared separately without affecting the core Falcon repository.
