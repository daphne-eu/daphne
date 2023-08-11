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

# Pull Request (PR) Guideline

## Terminology

- *Contributor*: the person who wants to contribute code by opening a PR
- *Collaborator*: a person who has the right to merge a branch into main (besides other rights) (official collaborator status on GitHub)
- *Reviewer*: a person who provides feedback on the contribution

## Disclaimer

**These guidelines are mainly for DAPHNE *collaborators***.
However, they could also be interesting for *contributors* to (a) understand how we handle PRs, and (b) learn which things we check, so they can try to prepare their contribution to speed up the review/merge procedure.

**Feel free to suggest changes to these guidelines** by opening an issue or a pull request if you feel something is missing, could be improved, needs further clarification, etc.

## Goals of these Guidelines

- **Merge useful contributions** (not necessarily perfect ones) into main without unnecessary delay
- **Guarantee a certain quality level**, especially since many people are working with the main branch
- **Balance the load** for handling PRs among collaborators

## PR Review/Merging Procedure

### PR creation

The contributor creates the PR.

- if the PR is marked as a draft, it is handled by an informal discussion depending on concrete questions by the contributor; if there are none, the PR is left alone for now
- if the PR is not marked as a draft, the review/merge procedure continues

### Initial response and reviewer assignment

The DAPHNE collaborators provide an *initial response* and *assign one (or multiple) reviewers* (usually from among themselves, but can also be non-collaborators).

- **Initial response**
  - ideally within a few working days after the PR was opened
  - thank contributor for the contribution
  - have a quick glance to decide if the contribution is relevant (default: yes)
- **Reviewer selection**
  - any collaborator (or non-collaborator) may volunteer
  - collaborators may select reviewer in a discussion (use @mentions)
  - who qualifies as a reviewer
    - collaborator experienced with the respective part of the code
    - collaborator mentoring the contributor (e.g., in case of undergrad students)
    - collaborator who wants to learn more about the respective part of the code base
    - any other collaborator to balance the reviewing load among collaborators
  - there may be multiple reviewers
- **Assignment of the reviewer(s)**
  - on GitHub
  - ideally, reviewer(s) should *communicate when review can be expected* (based on their availability and urgency of the PR)

### Rounds of feedback and response

If necessary, the reviewer(s) and the contributor prepare the contribution for a merge by multiple (but ideally not more than one) rounds of feedback and response.

**Reviewer examines the contribution:**

- **read the code, look at the diff**
    - *level of detail*
        - focus on integration into overall code base
        - if really special topic, which reviewer is not familiar with, then no deep review possible/required
        - especially for "good first issues": read code in detail
        - be the stricter the more central the code is (the more people are affected by it)
    - *clarify relation of PR to issue*
        - if PR states to address an issue, check if it really does so
        - it can be okay if a PR addresses just a part of a complex issue (if contribution still makes sense)
        - briefly check if PR addresses further issues (if so, also mention that in feedback and commit message later)
        - PR does not need to address an issue, but if it doesn't, check if contribution really belongs to the DAPHNE system itself (there might be useful contributions which should better reside in a separate repo, e.g., for the usage of DAPHNE, tools around DAPHNE, experiments/reproducibility, ...)
    - *contribution DOs*
        - readable code
        - necessary API changes should be reflected in the documentation (e.g., DaphneDSL/DaphneLib, command line arguments, environment variables, ...)
        - appropriate test cases (should be present and make sense, test expected cases, corner cases, and exceptional cases)
        - comments/explanations in central pieces of the code
        - meaningful placement of the contributed code in the directory hierarchy
        - correct integration of additional third-party code (reasonable placement in directory hierarchy, license compatibility, ...)
        - DAPHNE license header (*should be checked automatically*)
        - ...
    - *contributions DON'Ts*
        - obvious bugs (also think of corner cases)
        - changes unrelated to the PR (should be addressed in separate PRs)
        - significant performance degradation (in terms of building as well as executing DAPHNE) (*such checks should be automated*)
        - files that should not be committed, because they are not useful to others, too large, or can be generated from other files (e.g., IDE project files, output logs, executables, container images, empty files, unrelated files, experimental results, diagrams, unused files, auto-generated files, ...)
        - *unnecessary* API changes (e.g., DaphneDSL/DaphneLib, command line arguments, possibly environment variables, ...)
        - reimplementation of things we already have or that should better be imported from some third-party library
        - breaking existing code, formatting, tests, documentation, etc.
        - confidential information (usernames, passwords, ...)
        - paths on local system of contributor
        - misleading comments
        - copy-paste errors
        - extreme code duplication
        - useless prints (might even fail test cases)
        - whitespace changes that unnecessarily blow up the diff (especially in files that otherwise have no changes)
        - ...
    - *code style*
        - don't be strict as long as we don't have a clearly defined code style which can be enforced automatically
        - but watch out for things that make code hard to read, e.g.
            - wrong indentation
            - lots of commented out lines (especially artifacts from development/debugging)
- **try out the code**
    - check out the branch
        - If the contribution originates from a github fork, these steps will help to clone the PR's state into a branch of your working copy (example taken from PR #415):
            - Make sure your local copy of the main branch is up to date

                ```bash
                git checkout main
                git pull
                ```

            - Create a branch for the PR changes and pull them on top of that local branch

                ```bash
                git checkout -b akroviakov-415-densemat-strings-kernels main
                git pull git@github.com:akroviakov/daphne.git 415-densemat-strings-kernels
                ```

            - Once you have resolved all potential merge conflicts, you will have to do a merge commit. To get rid of this and ensure a linear history, start an interactive rebase from the last commit in main. In that process all non-relevant commits can be squashed and meaningful commit messages created if necessary.

                ```bash
                git rebase -i <commit hash of last commit in main> 
                ```

            - Once everything is cleaned up in the local PR branch, switch back to main and merge from the PR branch. This should yield clean commits on top of main because of the prior rebasing.

                ```bash
                git checkout main
                git merge akroviakov-415-densemat-strings-kernels
                git push origin main
                ```

    - check if the code builds at all (should be checked automatically)
    - check if there are compiler warnings (should be fixed) (should be checked automatically)
    - check if the test cases pass (should be checked automatically)
    - whether these checks succeed or fail may be platform-specific
        - **TODO:** think about that aspect in more detail

**Reviewer fixes minor problems:**

- things that are quicker to fix, than to communicate back and forth
    - typos and grammar mistakes (in variable names, status/error messages, comments, ...)
    - obvious minor bugs
    - wording/terminology (especially in comments)
- add separate commit(s) on PR's branch
    - to clearly separate these amendments from original contribution
- changes should be briefly mentioned/summarized in feedback
    - to document that something was changed
    - to notify contributor (ideally, they look at the changes in detail, learn from them, and do it better the next time)
- may be done at any point in time (before or after requested changes have been addressed by contributor)

**Reviewer provides feedback on the contribution:**

- **identify requests for concrete changes from contributor**
    - *things that the reviewer cannot fix within a few minutes*
        - more general corrections, refactoring, ...
        - more difficult bugs
    - *suitable for requesting mandatory changes*
        - in general, things that must to be done before the contribution can be merged, because there will be problems of some kind otherwise
        - bugs (functional, non-functional/performance)
        - things that could hinder others (e.g., unsolicited refactoring)
        - simplifications that make the code dramatically shorter and/or easier to read/maintain, and are straightforward to achieve
        - potentially also things that are in conflict with upcoming other PRs
    - *not suitable for requesting mandatory changes*
        - nice-to-have extensions of the feature: anything that could be done in a separate PR without leaving the code base in a bad state should not be a requirement for merging in at least a meaningful part of a feature
        - the contribution of a PR does not need to be perfect, but it should bring us forward
        - requests based on personal opinions which cannot be convincingly justified (e.g., implementing a feature in a different way as a matter of taste) (but might be okay for consistency)
        - top efficiency
        - such points can become follow-up issues and/or todos in the code (feel free to include issue number in todo)
- **reviewer gives feedback by commenting on the PR**
    - use the form on GitHub ("Files changed"-tab -> "Review changes": select "Approve" or "Request changes")
    - things to change should be enumerated clearly in the feedback on the PR (ideally numbered list or bullet points)
        - briefly explain why these requested changes are necessary
        - ideally provide some rough hints on how they could be addressed (but contributor is responsible for figuring out the details)
    - optional extensions can be added as suggestions (some contributors are very eager), but clearly say that they are not required before merging
    - feedback should be polite, actionable, concrete, and constructive

**Contributor addresses reviewer comments:**

- ideally, the contributor is willing to do this
- otherwise (and especially for new contributors, for whom we want to lower the barrier of entry), the reviewer or someone else should take charge of this, if possible

### Once the contribution is ready, a collaborator merges the PR

- can be done by the reviewer or any collaborator
- we want to keep a clean history on the main branch (and remember never to force-push to main)
    - makes it easier for others to keep track of the changes that happen
    - PR's branch might have untidy history with lots of commits for implementing the contribution and addressing reviewer comments; that should not end up on main
- typically, we want to rebase the PR branch on main, which may require resolving conflicts
- an example of how to use git on the command line is given in **try out the code in section 3.1** above
- **case A) if PR is conceptually one contribution**
    - on GitHub:
        - "Conversation"-tab: use "Squash and merge"-button (select this mode if necessary)
    - on the command line:
        - rebase and squash as required in the locally checked out PR branch
        - force-push to the PR branch (but never force-push to main)
        - locally switch to main and merge the PR branch
        - push to main
        - note: this procedure also ensures that the PR is shown as *merged* (not as *closed*) in GitHub later
    - *this will place a single new commit onto main* because you rebased/squashed in the PR branch first
- **case B) if PR consists of individual meaningful commits of a larger feature (with meaningful commit messages)**
    - on GitHub:
        - "Conversation"-tab: use "Rebase and merge"-button (select this mode if necessary)
    - on the command line:
        - rebase as required in the locally checked out PR branch
        - force-push to the PR branch (but never force-push to main)
        - locally switch to main and merge the PR branch
        - push to main
        - note: this procedure also ensures that the PR is shown as *merged* (not as *closed*) in GitHub later
    - *this will place the new commits onto main* because you rebased in the PR branch first
- **in any case**
    - enter a meaningful commit message (what and why, closed issues, ...)
        - ideally reuse the initial description of the PR
        - in case of squashing (case A above): please remove the unnecessarily long generated commit message)
        - **TODO:** commit messages should be a separate item in the developer documentation
    - *authorship*
        - if multiple authors edited the branch: choose one of them as the main author (after squashing in GitHub it should be the person who opened the PR); more authors can be added by adding [`Co-authored-by:  NAME NAME@EXAMPLE.COM`](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-with-multiple-authors) after two blank lines at the end of the commit message (one for each co-author).
        - very often, reviewers may have made minor fixes, but should refrain from adding themselves as co-authors (prefer to give full credit for the contribution to the initial contributor, unless the reviewer's contribution was significant)

### Creation of follow-up issues (optional)

- things that were left out
- nice-to-haves
- functional, non-functional, documentation, tests

### Inviting the contributor as a collaborator (conditional)

If this contributor has made enough non-trivial contributions of good quality (currently, we require three), he/she should be invited as a collaborator on GitHub.

## More Hints

- reviewer's time is precious, don't hesitate to request changes from the contributor (but keep in mind that we want to lower the barrier of entry for new contributors)
- avoid making a PR too large
    - makes it difficult to context-switch into it again and again
    - makes overall changes hard to understand and diffs hard to read
- whenever in doubt: use discussion features on GitHub to get others' opinions

## Communication

We want to facilitate an open and inclusive atmosphere, which should be reflected in the way we communicate.

- **TODO:** we should set up concrete guidelines for that, but that's actually a separate topic

- always be polite and respectful to others
- keep the conversation constructive
- keep in mind the background of other persons
    - experienced DAPHNE collaborator or new contributor
    - level of technical experience
    - English language skills
- ...
