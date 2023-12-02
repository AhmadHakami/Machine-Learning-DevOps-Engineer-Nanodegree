
# Github & Git

This repository serves as a guide for understanding the basics of Git and GitHub.

![Git vs Github](https://miro.medium.com/v2/resize:fit:1400/1*tDz-Vkeg-yoBRcnAZ5SDow.png)

---

## Clone a Repository from Github

```bash
git clone <repository-link.git>
```

Cloning a repository creates a local copy on your machine, allowing you to work on the project independently.

## Create a File Using CMD

```bash
touch <file-name.format>
```

Example:

```bash
touch pipeline.py
```

Creating a file is the first step in adding new content to your repository.

## Open and Edit a File

```bash
nano <file-name>
```

Using a text editor like nano helps you make updates or alterations to your files directly from the terminal.

## Check Files in the Directory

```bash
ls
```

Listing the files in your directory ensures you know what contents are currently available.

## Check Changes Inside the Repository

```bash
git status
```

Checking the status of your repository can show you which changes are staged, which are not, and which files are not being tracked by Git.

## Approve Changes to Add Them

```bash
git add <file-name>
```

or

```bash
git add .
```

Adding changes readies them to be committed to your local repository.

## Commit Changes

```bash
git commit -m "Commit message"
```

## Push Repository to GitHub

```bash
git push origin <branch-name>
```

Pushing your commits to GitHub updates the remote repository with your local changes.

## Connecting GitHub Account with Terminal

To connect your GitHub account, you may need to provide your credentials, which can be your username and password or a personal access token if two-factor authentication is enabled.

1. Set your username:

   ```bash
   git config --global user.name "your_github_username"
   ```

2. Set your email:

   ```bash
   git config --global user.email "your_email@example.com"
   ```

3. Authenticate with GitHub from the terminal when prompted or by setting up a credential helper that stores your GitHub credentials.

By configuring your GitHub account connection, you assure that all the contributions you make are attributed to you, and the code push or pull requests come from an authorized source.

## Check Current Branch

```bash
git branch
```

## Create new branch

```bash
git branch <branch name>
```

## Create and switch to a new branch

```bash
git checkout -b <branch-name>
```

## Remove branch

##### requires that you are not currently on the branch you would like to delete

```bash
git branch -d <branch-name>
```

## Pull the latest changes on the Branch

```bash
git pull
```

## View the log history

```bash
git log
```

## Read More:

1. [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
2. [About merge conflicts](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/about-merge-conflicts)
3. [How to Version Control Your Production Machine Learning Models](https://www.datarobot.com/blog/how-to-version-control-your-production-machine-learning-models/)
4. [Version control ML model](https://towardsdatascience.com/version-control-ml-model-4adb2db5f87c)
5. [Code Review Best Practices](https://www.kevinlondon.com/2015/05/05/code-review-best-practices)