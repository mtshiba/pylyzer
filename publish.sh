os=$(uname -s)
if [ "$os" = "Darwin" ]; then
    platform="macos"
elif [ "$os" = "Linux" ]; then
    platform="linux"
else
    echo "Unsupported platform: $os"
    exit 1
fi
cibuildwheel --output-dir dist --platform $platform
twine upload -u mtshiba -p $PYPI_PASSWORD --skip-existing dist/*
