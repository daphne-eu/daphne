# Contributing to the DAPHNE Prototype

*Thank you* for your interest in contributing to the DAPHNE Prototype.
This is a team-effort that can only be successful if many people contribute to it.

Below you find a couple of *guidelines* (rather than rules) for contributing.
Since the Prototype repository is still in an early stage, this document is also *likely to be extended* step by step.

-----

There are various **ways of contributing** including (but not limited to):
- implementation work
- writing test cases
- writing documentation
- reporting bugs or any other kind of issue
- contributing to discussions

We encourage **open communication** about the code through issues and comments here on GitLab.
That way, we can document the discussion, make it accessible to all partners, and avoid repetition in case someone else has the same question/comment.

### Contributing to the source code

**Issue tracking**

All open/ongoing/completed **work is tracked as issues** here on GitLab.
These could be anything from precisely defined small tasks to requests for complex components.
In any case, we will try to keep the book-keeping effort at a low level; at the same time, we should make each other aware of what everyone is working on.

If you would like to contribute and are **looking for a task** to work on, browse the [list of issues](https://gitlab.know-center.tugraz.at/daphne/prototype/-/issues).
If you are new to the prototype, you might want to watch out for the label ["good first issue"](https://gitlab.know-center.tugraz.at/daphne/prototype/-/issues?label_name%5B%5D=good+first+issue).
Once you have selected an issue to work on, please assign yourself or leave a comment.
It should be explicit who is working on which issue to avoid duplicate work by different people.

Furthermore, you are invited to **create issues yourself**, e.g., for tasks you want to work on or problems you encountered.
This is also a good way to enable discussion on the topic.
Note that there is a set of labels that can be attached to your issue to clarify what it is about and to make it more easy to find.

**Contribution workflow**

To contribute to the source code, please clone the repository, create a new branch, develop your changes in that branch, and finally make a merge request (aka pull request) to have your changes integrated into the master branch.
Please note that you can create as many branches and merge requests as you want, it's a really lightweight thing.

- *Cloning the repository.*

  `git clone --recursive https://gitlab.know-center.tugraz.at/daphne/prototype.git`, as described in [Getting Started](https://gitlab.know-center.tugraz.at/daphne/prototype/-/blob/master/doc/GettingStarted.md).

- *Working on your own branch.*
  
  Start by creating a new branch locally: `git checkout -b BRANCH_NAME`.
  Please select `BRANCH_NAME` to clearly indicate what it is about or who is responsible for it, ideally by referring to the issue, e.g. `dev-issue-123`.
  Make as many commits to your branch as you like.
  When you push the branch for the first time, use `git push --set-upstream origin BRANCH_NAME`.
  Finally, note that to push to the repository, you need to have the developer role.
  To get promoted to a developer, please assign yourself to an issue or contact @pdamme.

- *Integrating your branch into the master branch.*

  Once you feel that your contribution is ready to be merged into the master branch, please create a merge request in GitLab.
  *Details on this will be added soon.*
  You can also create a merge request before the contribution is ready if you would like to discuss about it.
  In that case, please prefix the merge request name with "[WIP]".
  After your branch has been merged, it can be deleted.

**Code style**

This a topic on its own.
We are going to define a code style soon.
Until then, please try to be consistent with the code that is already there.
