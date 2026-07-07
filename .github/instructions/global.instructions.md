# Language

- Always respond in Vietnamese.
- Only respond in English when my prompt starts with `English:`.

# File Modification Rules

- By default, do **not** modify any existing files.
- Only modify existing files when my prompt starts with `Change:`.
- When modifying files, make only the changes necessary to fulfill the request. Do not make unrelated changes.

# New Files

- Only create new files when my prompt starts with `New:`.
- Do not create new files unless explicitly requested with the `New:` prefix.

# Suggestions

- If my prompt does not start with `Change:` or `New:`, you may:
  - Explain code.
  - Analyze bugs or errors.
  - Suggest improvements or alternative approaches.
  - Provide example code snippets in your response.
- Do not apply any changes directly to the workspace or modify/create any files.