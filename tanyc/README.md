
## git 工作流
```bash
# --- 创建项目分支 ---
git checkout main
git pull upstream main
git checkout -b project/xxx

# --- 每次开发前 ---
git checkout main
git pull upstream main
git checkout project/xxx
git rebase main

# --- 日常开发 ---
# 修改代码
git add .
git commit -m "xxx"
git push origin project/xxx

# --- 阶段性结束 ---
# GitHub 发 PR: project/xxx → origin/main

# --- 合并后 ---
git checkout main
git pull upstream main
git checkout -b project/new_xxx
```
