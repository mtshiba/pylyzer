if [[ "$PWD" == */pylyzer ]]; then
    if [ "$1" != "--pip-only" ]; then
        cd crates/py2erg
        echo "publish py2erg ..."
        cargo publish
        cd ../../
        cargo publish
        if [ "$1" = "--cargo-only" ]; then
            exit 0
        fi
    fi
    if [ "$1" != "--cargo-only" ]; then
        if [ "$PYPI_PASSWORD" = "" ]; then
            echo "set PYPI_PASSWORD"
            exit 1
        fi
        maturin build --release
        ver=`cat Cargo.toml | rg "^version =" | sed -r 's/^version = "(.*)"/\1/'`
        whl=target/wheels/`ls target/wheels | rg "pylyzer-$ver"`
        python3 -m twine upload $whl -u mtshiba -p $PYPI_PASSWORD
    fi
    echo "completed"
else
    echo "use this command in the project root"
fi
