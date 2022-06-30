====
EIS Toolkit
====

**For developers**

In case you add a new class or function into the toolkit, please update the documentation site!

1. Modify mkgendocs.yml by adding a new page to pages section

- Give name to a new page, e.g. new_class.md
- Give path to the corresponding python file, e.g. eis_toolkit/new_class.py
- Give list of the functions to be documented

2. Run gendocs --config mkgendocs.yml

3. Run mkdocs serve

4. Go to http://127.0.0.1:8000/ to see the documentation
