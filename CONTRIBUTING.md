# Contributing to Voxelizer-rs

Thank you for your interest in contributing to Voxelizer-rs! Please review the guidelines below to help keep the codebase maintainable, performant, and secure.

## Workflow

1. Read this document and the `readme.MD` to understand the project's purpose and functionality.
2. Read the codebase to familiarize yourself with the structure and capabilities.
3. Analyze possible improvements. Focus on one impactful feature or fix at a time.
4. Implement your changes.
5. Run tests and linters locally before submitting. Ask for clarification if needed.

## Pull Request Guidelines

To help us categorize and review changes quickly, please use the following emojis and structure in your Pull Request titles and descriptions.

### PR Titles
* `🔒 [security fix description]` - Security patches.
* `🧹 [code health improvement description]` - Refactors, formatting, dead-code removal, documentation.
* `⚡ [performance improvement description]` - Optimizations.
* `🧪 [testing improvement description]` - New tests or test framework upgrades.

### PR Descriptions
Structure your PR description based on the type of change:

* **Security Fixes (`🔒`)**:
  * 🎯 **What**: Describe the vulnerability.
  * ⚠️ **Risk**: What is the impact?
  * 🛡️ **Solution**: How does this PR fix it?
* **Code Health (`🧹`)**:
  * 🎯 **What**: Describe the refactor or cleanup.
  * 💡 **Why**: Why is this better?
  * ✅ **Verification**: How did you verify it didn't break functionality?
  * ✨ **Result**: What is the net improvement?
* **Performance (`⚡`)**:
  * 💡 **What**: Describe the optimization.
  * 🎯 **Why**: Why was it slow?
  * 📊 **Measured Improvement**: Before vs. After metrics.
* **Testing (`🧪`)**:
  * 🎯 **What**: Describe the new tests.
  * 📊 **Coverage**: What new areas are covered?
  * ✨ **Result**: Why does this make the project safer?

## Coding Standards & Testing

Voxelizer-rs relies heavily on `rayon` for parallelization and `parry3d` for 3D geometry.

* **Formatting**: We strictly enforce `rustfmt`. Run `cargo fmt --all` before committing.
* **Linting**: We enforce zero warnings. Run `cargo clippy -- -D warnings` before committing. If pre-existing complex return types trigger warnings in unchanged code, use `#[allow(clippy::type_complexity)]` rather than refactoring unnecessarily.
* **Testing**: Run the full test suite with `cargo test`.
  * When writing tests for grid-based processing (e.g., voxelization), use coarse resolutions (e.g., 0.5) for assertions to prevent test timeouts ($N^3$ scaling).
  * Use explicit assertions. For error paths, verify that `Result::Err` is returned and check the exact error message string.
* **Hygiene**: Remove all temporary exploration files (`.patch`, `.orig`, test binaries, etc.) before finalizing commits so they do not end up in the PR.
* **Dependencies**: Do not add new dependencies or modify manifest files without explicit permission.

## Submitting

Once your code is ready:
1. Ensure `cargo fmt --all -- --check` passes.
2. Ensure `cargo clippy -- -D warnings` passes.
3. Ensure `cargo test` passes.
4. Push your branch and open a PR following the template guidelines above.