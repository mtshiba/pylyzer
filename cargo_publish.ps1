if ($PWD.Path -like "*\pylyzer") {
    if ($null -eq $env:PYPI_PASSWORD) {
        echo "set PYPI_PASSWORD environment variable"
        exit
    }
    if ($args[0] -ne "--pip-only") {
        cd crates/py2erg
        echo "publish py2erg ..."
        cargo publish
        # from cargo 1.66 timeout is not needed
        # timeout 12
        cd ../../
        cargo publish
    }
    maturin build --release
    $ver = cat Cargo.toml | rg "^version =" | sed -r 's/^version = "(.*)"/\1/'
    $whl = "target/wheels/$(ls target/wheels | Select-Object -ExpandProperty Name | rg "pylyzer-$ver")"
    python -m twine upload $whl -u mtshiba -p $env:PYPI_PASSWORD
    echo "completed"
} else {
    echo "use this command in the project root"
}
