from fnmatch import fnmatch


ignore_patterns = [
    '**/node_modules/**', '**/__pycache__/**', '**/*.pyc', '**/.idea/**', '**/.vscode/**', '**/venv/**',
    '**/.env/**', '**/.git/**', '**/dist/**', '**/build/**', '**/.DS_Store', '**/*.log', '**/tmp/**',
    '**/.npm/**', '**/coverage/**', '**/.nyc_output/**', '**/.cache/**', '**/.next/**', '**/out/**',
    '**/.parcel-cache/**', '**/public/build/**', '**/.webpack/**', '**/lib/**', '**/target/**',
    '**/Cargo.lock', '**/*.class', '**/*.jar', '**/.gradle/**', '**/build.gradle', '**/settings.gradle',
    '**/*.iml', '**/out/**', '**/.rustup/**', '**/.cargo/**', '**/__sapper__/**', '**/.svelte-kit/**',
    '**/svelte-kit/**', '**/.vercel/**', '**/vercel/**', '**/.venv/**', '**/.git/**', '**/__init__.py',
    '**/tree-sitter**', '**/myenv/**', '**/*.min.js', '**/*.config.js', '**/META-INF/**', '**/*.class', '**/*.jar',
    "*.so", '**/*.bundle.js', '.git/**', '**/*.svg', '**/*.png', '**/*.jpeg', '**/*.jpg', '**/*.gif', '**/*.mp4',
    '**/*.csv', '**/*package-lock.json', '**/*.gitignore'
]


def is_ignored(file_path):
    """
    Checks if the file is ignored
    :param file_path:
    :param ignorables:
    :return:
    """
    for pattern in ignore_patterns:
        if fnmatch(file_path, pattern):
            return True
    return False
