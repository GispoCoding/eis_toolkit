# Generating documentation
In case you add a new class, module or function into the toolkit, please update the documentation site!

1. Modify mkgendocs.yml by adding a new page to pages section:
  - Give name to a new page, e.g. geoprocess/clip.md
  - Give path to the corresponding python file, e.g. eis_toolkit/geoprocess/clipping.py
  - Give list of the function names to be documented, e.g. clip

2. Navigate to the root directory level (the same level where mkgendocs.yml file is located) and run

```shell
gendocs --config mkgendocs.yml
```

*Executing the command above automatically creates new (empty) version of the index.md file. However, this is not desired behaviour since the index.md file already contains some general information about the eis_toolkit. Hence, please use Rollback or otherwise undo the modifications in index.md file before committing, or do not commit the index.md file at all.*

3. Run

```shell
mkdocs serve
```

4. Go to http://127.0.0.1:8000/

If you **just** want to take a look at the documentation (not to modify it), just execute steps 3 and 4.
