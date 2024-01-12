# study
## github 명령어
* 진행순서
```
$ git remote origin add (repository 주소)
$ git status
$ git branch
$ git checkout main
$ git add .
$ git commit -m "create_back_project"
$ git push origin main

$ git pull origin main

$ git fetch --all
$ git reset --hard origin/main

$ git pull --rebase origin main
-> main 으로 강제 변경
$ git rebase --abort

$ pip freeze > requirements.txt # 사용 라이브러리 설치파일 리스트 만들기
$ pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

* --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org # ssl 방화벽 해제 후 설치
```
* 예외사항 시
```
$ git reset "HEAD^" #가장 최신 커밋 취소
$ git reset "HEAD~3" #최신 커밋 3개 취소

$ git push origin master -f #master brench에 강제로 push

$ git push -u origin 브랜치명 --force  :  강제로 브랜치, 마스터에 업로드 
$ git log : 기록 보면서
$ git reset HEAD^ : 최신 상태에서 바로 전 상태로 롤백하는 명령어
```
