# Import Audit After Refactor

After refactoring modules, verify all import sites are updated correctly.

## Use Cases

- After splitting a module (e.g., `config.py` → `config/` package)
- After renaming classes or functions
- After moving code between modules

## Steps

1. **Find all import sites:**
   - Use `grep -r` to find old import patterns
   - Check `src/`, `tests/`, `scripts/`
   - Include both production and test code

2. **Verify updates:**
   - Ensure imports resolve to new locations
   - Check for forgotten import sites
   - Verify test imports updated

3. **Test import resolution:**
   - Run quick Python import check
   - Run smoke tests to catch import errors

4. **Check for broken code:**
   - Look for code that references old module paths
   - Check docstrings and comments for stale references

## Expected Output

- List of all import sites found
- Confirmation that all are updated
- Results of import resolution test
- Any remaining issues to fix
