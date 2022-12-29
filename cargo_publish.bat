@echo off

if %~dp0 == C:%homepath%\GitHub\pylyzer\ (
    cd crates/py2erg
    echo publish py2erg ...
    cargo publish
    rem from cargo 1.66 timeout is not needed
    rem timeout 12
    cd ../../
    cargo publish
    echo completed
) else (
    echo Use this command in the project root
)
