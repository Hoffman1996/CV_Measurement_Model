# Git quick‑guide (Windows/PowerShell + VS Code)
0) Check where you are & what’s going on

## where am I and what’s changed?
git status

## which branch, and is it ahead/behind?
git branch -vv

## what commits are on my machine but not on GitHub yet?
git log origin/main..HEAD --oneline

## what remotes are configured?
git remote -v

---

1) Keep your local repo up to date

## fetch latest refs (no file changes yet)
git fetch origin

## merge remote main into your local main
git pull origin main
(or just `git pull` if your upstream is set)

---

2) Add or replace files/folders
Replacing is the same as adding: overwrite the file(s), then stage and commit.

## add specific files
git add path\to\file1.ext path\to\file2.ext

## add everything new/changed in the current folder
git add .

## commit with a message (skip the editor)
git commit -m "Add new dataset split and update training script"

## push to GitHub
git push
Paths with spaces: wrap in quotes, e.g.
git add "DECISION MAKING AND REASONING.txt"

---

3) Remove files/folders

## remove tracked files (and delete them from disk)
git rm path\to\file.ext
git commit -m "Remove obsolete artifact"
git push

## remove a folder (recursively)
git rm -r path\to\folder
git commit -m "Remove old training outputs"
git push

---

4) Rename / move files (keeps history)

git mv old\path\name.ext new\path\name.ext
git commit -m "Rename/move file"
git push

---

5) .gitignore tips (and “why is Git still tracking this?”)

## Add ignore rules in .gitignore (one per line), e.g.:
venv/
datasets/
archive/
runs/
yolo_training_output/
*.zip
*.pt
*.ckpt

## If a file was already committed before you added it to .gitignore, Git will keep tracking it. To stop tracking without deleting your working copy:
git rm -r --cached .
git add .
git commit -m "Refresh .gitignore (stop tracking ignored files)"
git push
(This clears the index, re-adds only non-ignored files, and preserves anything ignored on disk.)

---

6) See what folders are tracked (only directories)

## show only tracked folders at HEAD
git ls-tree -r --name-only HEAD | % { Split-Path $_ -Parent } | Sort-Object -Unique

---

7) Abort/escape common “stuck” states

## you opened a merge by mistake?
git merge --abort

## you started a rebase?
git rebase --abort

## you’re in the middle of an in-progress commit editor in terminal:
##   close the editor OR commit with a message next time:
git commit -m "Your message"

## Make VS Code your Git editor (so no vim pops up)
git config --global core.editor "code --wait"

---

8) Undo mistakes (safe patterns)

## undo staged changes back to “modified”
git reset HEAD path\to\file.ext

## throw away local changes in a file (revert to last commit)
git checkout -- path\to\file.ext

## (or in newer git)
git restore path\to\file.ext

## undo the last commit but keep the changes staged
git reset --soft HEAD~1

## undo the last commit and unstage the changes (keep in working tree)
git reset --mixed HEAD~1

## hard reset (⚠️ discards local changes!)
git reset --hard HEAD~1

---

9) Create a new branch (optional but good practice)

git checkout -b feature/auto-label-florence

## work, add, commit…
git push -u origin feature/auto-label-florence

## open PR on GitHub when ready

---

10) Useful logs & diffs

## compact history
git log --oneline --graph --decorate --all

## what changed in this commit
git show HEAD

## diff working tree vs last commit
git diff

## diff what’s staged vs last commit
git diff --cached

---

11) Large files & line endings (nice-to-haves)

## Git LFS for big artifacts (models, zips):
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add *.pt
git commit -m "Track model weights with LFS"
git push

## Line endings: ensure consistent behavior across machines
git config --global core.autocrlf true   # on Windows

---

12) End‑to‑end examples (your recent patterns)

## Replace .docx with .txt
git add "DECISION MAKING AND REASONING.txt" summary.txt
git rm  "DECISION MAKING AND REASONING.docx" summary.docx
git commit -m "Replace .docx with .txt narratives"
git push

## Remove old YOLO outputs
git rm -r yolo_training_output/
git commit -m "Remove stale training outputs"
git push

## Add new Florence‑labeled dataset
git add datasets\yolo_dataset\ -A
git commit -m "Add updated train/valid images and labels (Florence auto-label)"
git push