# Github & Git
This repository serves as a guide for understanding the basics of Git and GitHub.

![Git vs Github](https://miro.medium.com/v2/resize:fit:1400/1*tDz-Vkeg-yoBRcnAZ5SDow.png)
_________________
### Clone a Repository from Github

```git clone <repository-link.git>```

Cloning a repository creates a local copy on your machine, allowing you to work on the project independently.

### Create a File Using CMD

```touch <file-name.format>```
Example: ```touch pipeline.py```

Creating a file is the first step in adding new content to your repository.

### Open and Edit a File

```nano <file-name>```

Using a text editor like nano helps you make updates or alterations to your files directly from the terminal.

### Check Files in the Directory

```ls```

Listing the files in your directory ensures you know what contents are currently available.

### Check Changes Inside the Repository

```git status```

Checking the status of your repository can show you which changes are staged, which are not, and which files are not being tracked by Git.

### Approve Changes to Add Them

```git add <file-name>``` or ```git add .```

Adding changes readies them to be committed to your local repository.

### Commit Changes

```git commit -m "Commit message"```

### Push Repository to GitHub

```git push origin <branch-name>```

Pushing your commits to GitHub updates the remote repository with your local changes.

### Connecting GitHub Account with Terminal

To connect your GitHub account, you may need to provide your credentials, which can be your username and password or a personal access token if two-factor authentication is enabled.

1. Set your username:
   ```git config --global user.name "your_github_username"```

2. Set your email:
   ```git config --global user.email "your_email@example.com"```

3. Authenticate with GitHub from the terminal when prompted or by setting up a credential helper that stores your GitHub credentials.

By configuring your GitHub account connection, you assure that all the contributions you make are attributed to you, and the code push or pull requests come from an authorized source.


### Check Current Branch
```git branch```

### Create new branch
```git branch <branch name>```

### Create and switch to a new branch
```git checkout -b <branch-name>```

### Remove branch 
##### requires that you are not currently on the branch you would like to delete
```git branch -d <branch-name>```


### Pull the latest changes on the Branch
```git pull```

### View the log history
```git log```

## Sources:
1.  [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)

2. [About merge conflicts
](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/about-merge-conflicts)

3. [How to Version Control Your Production Machine Learning Models
](https://www.datarobot.com/blog/how-to-version-control-your-production-machine-learning-models/)

4. [Version control ML model](https://towardsdatascience.com/version-control-ml-model-4adb2db5f87c)

5. [Code Review Best Practices](https://www.kevinlondon.com/2015/05/05/code-review-best-practices)

6. [Code Review Best Practices
](https://www.kevinlondon.com/2015/05/05/code-review-best-practices)