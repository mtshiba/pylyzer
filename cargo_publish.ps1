if ($PWD.Path -eq "$HOME\GitHub\pylyzer") {
    cd crates/py2erg
    echo "publish py2erg ..."
    cargo publish
    # from cargo 1.66 timeout is not needed
    # timeout 12
    cd ../../
    cargo publish
    maturin build --release
    $ver = cat Cargo.toml | rg "^version =" | sed -r 's/^version = "(.*)"/\1/'
    python -m twine upload "target/wheels/pylyzer-$ver-py3-none-win_amd64.whl" -u mtshiba -p $env:PYPI_PASSWORD
    echo "completed"
} else {
    echo "use this command in the project root"
}
