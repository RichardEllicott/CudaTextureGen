# Git Commands

```powershell
# Add to repo process
git status # check what changes have been made
git add . # add all the change files
git status # look again, files will go green
git commit -m "Describe your changes here" # make a commit
git push origin windows-rebuild # finally push (to branch windows-rebuild)
```

```powershell
# ⚠️ Reset all changes to last commit
git reset --hard HEAD
git clean -fd
```