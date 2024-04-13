# Compiling pyAMPACT documentation

For a single-version build, we can use the Makefile in the `docs/` folder, e.g.:

```
make html
```
or
```
make latexpdf
```

For a multi-version historical build (i.e., our doc site), we need to use
sphinx-multiversion.

This can be done from the top-level source directory of the repository as follows:

```
sphinx-multiversion -D smv_latest_version=$(./scripts/get_latest_release.sh) docs/ build/html
```

This says that the source config lives in `docs/` and the output site will be
deposited under `build/html`.  To deploy, we sync the compiled site to the
`pyAMPACT-doc` repository, which in turn publishes to github-pages.