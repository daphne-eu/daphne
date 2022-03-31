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

If you would like to contribute and are **looking for a task** to work on, browse the [list of issues](https://github.com/daphne-eu/daphne/issues).
If you are new to the prototype, you might want to watch out for the label ["good first issue"](https://github.com/daphne-eu/daphne/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
Once you have selected an issue to work on, please assign yourself or leave a comment.
It should be explicit who is working on which issue to avoid duplicate work by different people.

Furthermore, you are invited to **create issues yourself**, e.g., for tasks you want to work on or problems you encountered.
This is also a good way to enable discussion on the topic.
Note that there is a set of labels that can be attached to your issue to clarify what it is about and to make it more easy to find.

**Contribution workflow**

To contribute to the source code, please clone the repository, create a new branch, develop your changes in that branch, and finally make a merge request (aka pull request) to have your changes integrated into the master branch.
Please note that you can create as many branches and merge requests as you want, it's a really lightweight thing.

- *Cloning the repository.*

  `git clone --recursive https://github.com/daphne-eu/daphne.git`, as described in [Getting Started](https://github.com/daphne-eu/daphne/blob/master/doc/GettingStarted.md).

- *Working on your own branch.*
  
  Start by creating a new branch locally: `git checkout -b BRANCH_NAME`.
  Please select `BRANCH_NAME` to clearly indicate what it is about or who is responsible for it, ideally using the pattern `123-some-short-title` (where 123 is the issue number).
  Make as many commits to your branch as you like.
  When you push the branch for the first time, use `git push --set-upstream origin BRANCH_NAME`.
  Finally, note that to push to the repository, you need to have the developer role.
  To get promoted to a developer, please assign yourself to an issue or contact @pdamme.

- *Integrating your branch into the master branch.*

  Once you feel comfortable with your changes, you can ask for their integration into the master branch by creating a *merge request*.
  Please choose an expressive title and provide a short description of your changes.
  The default merge options are to (1) squash the commits on your branch into a single commit on the master branch to keep a clean history, and (2) to delete your feature branch after a successful merge.
  However, these can be decided on a case-by-case basis, and even after the merge request has been created.

  After you have created the merge request, GitLab might tell you: "Fast-forward merge is not possible. To merge this request, first rebase locally"
  This happens when there are conflicts between your branch and the master branch.
  In that case, please try to resolve the conflicts yourself locally and push any new commits to your branch.
  If you do not feel comfortable with that, please ask for help by commenting on your merge request.

  Please also feel free to create a merge request to ask for feedback or to start a discusssion on the code.
  In that case, you should prefix the title of your merge request with "Draft:" or "WIP:" to indicate that you don't ask for its integration right now.

  Finally, you can always add more commits to your merge request by pushing to your branch.

**Code style**

This a topic on its own.
We are going to define a code style soon.
Until then, please try to be consistent with the code that is already there.
