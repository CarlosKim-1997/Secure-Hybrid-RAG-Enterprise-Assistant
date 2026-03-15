# Git 업로드 가이드 (RAGforCompany)

이 문서는 이 프로젝트를 GitHub 등 원격 저장소에 올리는 방법을 단계별로 안내합니다.

---

## 1. 준비 확인

### 1.1 Git 설치
```bash
git --version
```
- 미설치 시: [Git 다운로드](https://git-scm.com/downloads) 후 설치

### 1.2 올리면 안 되는 파일
- **`.env`** — API 키 등 비밀 정보 (이미 `.gitignore`에 포함됨)
- **`data/vectordb/`** — FAISS 인덱스 (재생성 가능하므로 제외)
- 가상환경(`.venv/`, `venv/`) — 제외됨

→ **`.env`는 절대 커밋하지 마세요.** `.env.example`만 올리면 됩니다.

---

## 2. 저장소 초기화 및 첫 커밋

프로젝트 루트(`RAGforCompany` 폴더)에서 실행하세요.

```bash
# 1) Git 저장소 초기화 (아직 안 했다면)
git init

# 2) 현재 상태 확인 (.env, vectordb 등 제외된 것 확인)
git status

# 3) 모든 파일 스테이징
git add .

# 4) 첫 커밋
git commit -m "Initial commit: Secure Hybrid RAG Enterprise Assistant PoC"
```

---

## 3. 원격 저장소 연결 및 푸시

### 3.1 GitHub에서 저장소 생성
1. [GitHub](https://github.com) 로그인
2. **New repository** 클릭
3. Repository name: `RAGforCompany` (또는 원하는 이름)
4. **Public** 선택, **README / .gitignore / License 추가하지 않음** (로컬에 이미 있음)
5. **Create repository** 클릭

### 3.2 로컬과 연결 후 푸시
GitHub에서 생성 후 나오는 주소(HTTPS 또는 SSH)를 사용합니다.

**HTTPS 예시:**
```bash
git remote add origin https://github.com/내계정/RAGforCompany.git
git branch -M main
git push -u origin main
```

**SSH 예시 (SSH 키 설정된 경우):**
```bash
git remote add origin git@github.com:내계정/RAGforCompany.git
git branch -M main
git push -u origin main
```

- `내계정` → 본인 GitHub 사용자명으로 변경
- 최초 푸시 시 GitHub 로그인 또는 토큰 입력 필요할 수 있음

---

## 4. 이후 작업 시 흐름

```bash
# 변경 사항 확인
git status

# 스테이징
git add .
# 또는 특정 파일만: git add app.py docs/README.md

# 커밋
git commit -m "작업 내용을 한 줄로 요약"

# 푸시
git push
```

---

## 5. 자주 쓰는 명령어

| 목적           | 명령어 |
|----------------|--------|
| 원격 주소 확인 | `git remote -v` |
| 브랜치 목록    | `git branch -a` |
| 최근 커밋 로그 | `git log --oneline -5` |
| 원격 최신 반영 | `git pull` |

---

## 6. 주의사항

- **API 키**: `.env`에 있는 `OPENAI_API_KEY`는 반드시 로컬에만 두고, GitHub에는 올리지 않습니다. 다른 사람이 클론해도 `.env`는 없으므로 `.env.example`을 복사해 본인 키를 넣어 사용하도록 README에 안내해 두었습니다.
- **대용량 파일**: `data/vectordb/`는 `.gitignore`에 있어 올라가지 않습니다. 클론한 뒤 `python init_vectordb.py`로 재생성하면 됩니다.
- **비공개 저장소**: 포트폴리오용으로 공개하고 싶지 않다면 GitHub에서 Repository 설정 → **Private**으로 두면 됩니다.

이 가이드대로 진행하면 저장소 초기화부터 푸시까지 한 번에 할 수 있습니다.
