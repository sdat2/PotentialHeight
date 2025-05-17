# How to create Sphinx docs

With the dev environment activated, and Sphinx installed you can create the html version by running the following command from this `docs` directory:

```bash
make html
```

And for the pdf version use

```bash
make latexpdf
```

Note: this last command requires a latex installation.

```bash
open _build/html/index.html 
```

## Other important commands

To update the module references in the rst files

```bash
sphinx-apidoc -f -o . ..
```


## Symbolic links

`docs/MAIN_README.md` 

is a symbolic link to items in the main directory.

This was done to trick sphinx into working, and seems to have worked so far.

## Lang.txt

To recreate the lang.txt count run this command from the main directory:

```bash 
git ls-files | cloc --report-file=docs/lang.txt --sum-one --exclude-ext=json,csv --list-file -
```