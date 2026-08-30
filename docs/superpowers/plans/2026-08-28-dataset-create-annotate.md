# Dataset Create and Annotate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable in-platform dataset creation via upload, class management, bbox/polygon annotation, batch ops, and snapshot generation without external tools.

**Architecture:** Extend `POST /api/datasets` empty branch to create `Dataset + LabelSchema` without snapshot; add `images`/`annotations`/`batch`/`snapshot` endpoints using `worker_session` and `BackgroundTasks`; frontend adds `DatasetCreatePage` + `DatasetAnnotatePage` with `AnnotateCanvas`.

**Tech Stack:** FastAPI, SQLAlchemy, React 18 + Vite, TypeScript, Canvas API, Pillow for validation

---

## File Structure

**Backend:**
- Modify: `backend/app/schemas/dataset.py` - Add new request/response schemas
- Modify: `backend/app/api/routes/datasets.py` - Add 4 new endpoints
- Modify: `backend/app/services/dataset_validation.py` - No change (already handles empty)
- New: `backend/tests/unit/test_dataset_annotate.py` - Unit tests
- New: `backend/tests/integration/api/test_dataset_annotate.py` - Integration tests

**Frontend:**
- New: `frontend/src/features/datasets/DatasetCreatePage.tsx`
- New: `frontend/src/features/datasets/DatasetAnnotatePage.tsx`
- New: `frontend/src/features/datasets/AnnotateCanvas.tsx`
- Modify: `frontend/src/app/routes.tsx` - Add routes
- Modify: `frontend/src/lib/api.ts` - Add ApiClient methods
- Modify: `frontend/src/lib/mockApi.ts` - Add mock branches
- Modify: `frontend/src/features/datasets/DatasetsPage.tsx` - Add create button link

---

### Task 1: Backend - Dataset Create Empty Branch

**Files:**
- Modify: `backend/app/api/routes/datasets.py:220-250`

- [ ] **Step 1: Write failing test**

```python
# backend/tests/unit/test_dataset_annotate.py
def test_create_empty_dataset():
    response = client.post("/api/datasets", json={"name": "empty-test", "source_path": None}, headers=headers)
    assert response.status_code == 201
    assert response.json()["name"] == "empty-test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -c NUL -p no:cacheprovider backend/tests/unit/test_dataset_annotate.py::test_create_empty_dataset -v`
Expected: FAIL

- [ ] **Step 3: Implement empty branch**

```python
# In create_dataset, if source_path is None, skip BackgroundTasks and just create Dataset
if not source_path_str:
    # Create empty dataset with default schema
    schema = LabelSchema(name=f"{name}_schema")
    db.add(schema)
    db.flush()
    # No snapshot, return early
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -c NUL -p no:cacheprovider backend/tests/unit/test_dataset_annotate.py::test_create_empty_dataset -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes/datasets.py backend/tests/unit/test_dataset_annotate.py
git commit -m "feat: empty dataset creation"
```

---

### Task 2: Backend - Image Upload Endpoint

**Files:**
- Modify: `backend/app/api/routes/datasets.py`
- Test: `backend/tests/integration/api/test_dataset_annotate.py`

- [ ] **Step 1: Write failing test**

```python
def test_upload_images():
    files = [("files", ("img.jpg", b"fake", "image/jpeg"))]
    response = client.post(f"/api/datasets/{dataset_id}/images", files=files, headers=headers)
    assert response.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -c NUL -p no:cacheprovider backend/tests/integration/api/test_dataset_annotate.py -k test_upload_images -v`
Expected: FAIL

- [ ] **Step 3: Implement endpoint**

```python
@router.post("/{dataset_id}/images")
def upload_images(dataset_id: UUID, files: list[UploadFile] = File(...)):
    for f in files:
        dest = Path(settings.dataset_dir) / "extracted" / str(dataset_id) / "images" / "train" / f.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as out:
            out.write(await f.read())
    return {"count": len(files)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -c NUL -p no:cacheprovider backend/tests/integration/api/test_dataset_annotate.py -k test_upload_images -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes/datasets.py backend/tests/integration/api/test_dataset_annotate.py
git commit -m "feat: dataset image upload"
```

---

### Task 3: Backend - Annotation Save Endpoint

**Files:**
- Modify: `backend/app/api/routes/datasets.py`
- Test: `backend/tests/integration/api/test_dataset_annotate.py`

- [ ] **Step 1: Write failing test**

```python
def test_save_annotation():
    payload = {"class_id": 0, "bbox": [0.5, 0.5, 0.2, 0.2]}
    response = client.put(f"/api/datasets/{dataset_id}/annotations/img_0001", json=payload, headers=headers)
    assert response.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -c NUL -p no:cacheprovider backend/tests/integration/api/test_dataset_annotate.py -k test_save_annotation -v`
Expected: FAIL

- [ ] **Step 3: Implement endpoint**

```python
@router.put("/{dataset_id}/annotations/{image_id}")
def save_annotation(dataset_id: UUID, image_id: str, payload: AnnotationPayload):
    # Validate class_id < nc, 0~1 range
    # Write to labels/train/{image_id}.txt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -c NUL -p no:cacheprovider backend/tests/integration/api/test_dataset_annotate.py -k test_save_annotation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes/datasets.py backend/tests/integration/api/test_dataset_annotate.py
git commit -m "feat: annotation save"
```

---

### Task 4: Frontend - Dataset Create Page

**Files:**
- Create: `frontend/src/features/datasets/DatasetCreatePage.tsx`
- Modify: `frontend/src/app/routes.tsx`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/lib/mockApi.ts`

- [ ] **Step 1: Write failing test**

```typescript
test('creates dataset', async () => {
  render(<DatasetCreatePage />);
  await user.type(screen.getByLabelText('名称'), 'test');
  await user.click(screen.getByText('创建'));
  expect(screen.getByText('创建成功')).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx playwright test`
Expected: FAIL

- [ ] **Step 3: Implement page**

Create `DatasetCreatePage.tsx` with form `名称 + 描述 + 初始类别列表`

- [ ] **Step 4: Run test to verify it passes**

Run: `npx playwright test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/features/datasets/DatasetCreatePage.tsx frontend/src/app/routes.tsx frontend/src/lib/api.ts frontend/src/lib/mockApi.ts
git commit -m "feat: dataset create page"
```

---

### Task 5: Frontend - Annotate Canvas

**Files:**
- Create: `frontend/src/features/datasets/AnnotateCanvas.tsx`
- Test: `frontend/tests/e2e/annotate.spec.ts`

- [ ] **Step 1: Write failing test**

```typescript
test('draws bbox', async ({ page }) => {
  await page.goto('/datasets/123/annotate');
  await page.mouse.move(100, 100);
  await page.mouse.down();
  await page.mouse.move(200, 200);
  await page.mouse.up();
  await expect(page.locator('.annotation')).toBeVisible();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx playwright test`
Expected: FAIL

- [ ] **Step 3: Implement canvas**

Create `AnnotateCanvas.tsx` with `bbox` drag, `polygon` point, `1-9` switch, `Del` delete

- [ ] **Step 4: Run test to verify it passes**

Run: `npx playwright test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/features/datasets/AnnotateCanvas.tsx frontend/tests/e2e/annotate.spec.ts
git commit -m "feat: annotate canvas"
```

---

### Task 6: Frontend - Datasets List Integration

**Files:**
- Modify: `frontend/src/features/datasets/DatasetsPage.tsx`

- [ ] **Step 1: Write failing test**

```typescript
test('shows create button', async ({ page }) => {
  await page.goto('/datasets');
  await expect(page.getByText('新建数据集')).toBeVisible();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx playwright test`
Expected: FAIL

- [ ] **Step 3: Implement link**

Add `+ 新建` button linking to `/datasets/create`

- [ ] **Step 4: Run test to verify it passes**

Run: `npx playwright test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/features/datasets/DatasetsPage.tsx
git commit -m "feat: datasets list create link"
```

---

## Verification

- `python -m pytest -c NUL -p no:cacheprovider backend/tests -q`
- `python backend/scripts/verify_bom.py`
- `cd frontend && npm run build`
- Manual: Create empty dataset → Upload → Annotate → Snapshot → Verify counts
