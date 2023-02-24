if [ "$PWD" = "$HOME/github/pylyzer" ]; then
    cd crates/py2erg
    echo "publish py2erg ..."
    cargo publish
    # from cargo 1.66 timeout is not needed
    # timeout 12
    cd ../../
    cargo publish
    maturin build --release
    ver=`cat Cargo.toml | rg "^version =" | sed -r 's/^version = "(.*)"/\1/'`
    whl=`ls target/wheels | rg "pylyzer-$ver"`
    python3 -m twine upload $whl -u mtshiba -p $PYPI_PASSWORD
    echo "completed"
else
    echo "use this command in the project root"
fi
