<!--
Copyright 2021 The DAPHNE Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Writing Documentation

At the moment the collection of markdown files in the `doc` directory is rendered to HTML and deployed via GitHub Pages.

If you insert a new markdown file, you have to add it into the html docs tree in [mkdocs.yml](/mkdocs.yml) at a suitable position under the `nav` section.

## Markdown Guideline

Please write clean markdown code to ensure a proper parsing by the tools used to render HTML. It is very recommended to use an IDE like *VS Code*. Code offers the feature to directly render markdown pages while you work on them. The extension [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint) directly highlights syntax violations / problems.

### Links

* With `[<link-name>](<link-url/path>)` you can link to other files in the repo
* Write links to other markdown files or source code files/diretories so that they work locally / in the github repository
* Do not use relative links like `../BuildingDaphne.md`
* Always use absolute paths relative to the repo root like `/doc/development/BuildingDaphne.md`
* The Links/URLs will be altered in order to work on the rendered HTML page as well
* Reference to issues with `[<description>](/issues/123)`. This won't work on github itself but will be rendered in the html page then

### Additional Syntax

While some markdown renderers are much more relaxed and render as wished, some points have to be considered so that mkdocs renders correctly as well.

* 4 spaces indentation for nested lists (ordered/unordered) and code blocks within lists to ensure proper rendering for HTML
* Using <\>: To use angle brackets use `<\>` notation outside an codeblock
    * Example: `<nicer dicer\>` renders to <nicer dicer\>

## Toolstack

* [MkDocs](https://www.mkdocs.org/) to build html from markdown files
* [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) as HTML theme
