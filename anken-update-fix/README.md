# Anken Update スキル修正パッチ (v2-11col / 2026-04-23)

## 概要
`月別サマリー.html` と `月別サマリー_フィルター付き.html` のスキーマが、Dashboard artifact「Anken Sales Dashboard」の契約スキーマと乖離していた問題の根本修正。

## 原因
`anken-update/SKILL.md` の Step 6-7「月別サマリーHTML再生成」セクションが疑似コードのみで、
具象コードが欠落していたため、毎回の定期実行で Claude が簡易版（9〜10列、売上データなし）を
生成し続けていた。

## 修正内容
SKILL.md の Step 6-7 を完全な具象コードに置き換える。

- **スキーマ契約**を明記（11列固定）
- **日付形式** `YYYY/MM/DD` で統一
- **小計行**に `style="background:#e2e6ea"` を付与（Dashboard のスキップ判定用）
- 同フォルダ `SKILL_step_6_7_patch.md` に置き換え対象のブロック全文

## 次のステップ
1. 同フォルダの `SKILL_step_6_7_patch.md` を開く
2. canonical な SKILL.md（OneDrive/GitHub 等）の Step 6-7 セクション全体をこの内容で置換
3. 変更を保存・同期
4. 次の定期実行（平日3時間ごと）で `月別サマリー.html` が11列スキーマで再生成される
5. Anken Sales Dashboard で再読込 → 正常動作を確認
