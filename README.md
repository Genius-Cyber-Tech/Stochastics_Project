# Stochastics_Project
Here is how to create the dev branch:

• git pull to make sure you are up to date.

• git switch -c dev to create the dev branch and switch to it.

• git push --set-upstream origin dev to connect your local dev branch with the repository.

Here is how to switch to the dev branch if it has already been created:

• git pull to make sure you are up to date.

• git branch -a to list all branches, you should see the dev branch.

• git fetch -p is only needed if you do not see the dev branch.

• git switch dev to work on the dev branch.

Now that you are working on the dev branch, upload the file to your repository on github. To do this, run the
following commands:

• git pull to make sure you are up to date.

• git status to see what has changed locally.

• git add . to add all changes (be careful to not upload the wrong files. If you want to upload only some files,
either change the parameters of the git add command or change the .gitignore file.)

• git status again to see what has been added with the last command.

• git commit -m "update members.txt" to commit.

• git push to upload your changes.

Now, merge the dev branch to the main branch and push to see the auto-testing in action. This push to main does
not count towards the maximum 3 pushes you are allowed to do per exercise.

• git pull to make sure you are up to date

• git switch main to switch from the dev to the main branch.

• git merge dev to merge dev into main. If you get merge conflicts, see e.g. this link on how to resolve them.
Basically you will have to change the files by hand and then add and commit them.

• git push to upload the merge.

After uploading, in your repository on github under Actions →your last commit →Autograding →education/au-
tograding , you can find out whether your members.txt file passed the test names test.

Note: You can have as many branches as you want. It can make sense to create branches for each person working
on the repository and merging them to dev, then merging to main at the end. However, you can also just all work
on dev at the same time and merge as needed.
