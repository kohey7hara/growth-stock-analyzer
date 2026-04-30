# SKILL.md Step 6-7 置換パッチ

**対象ファイル:** `anken-update/SKILL.md`
**置換範囲:** `#### 6-7: 月別サマリーHTML再生成` 見出しから、その直後の JavaScript コードブロック末尾 `})()` まで（およそ 671-723 行）

---

## 置換前（現状・削除対象）

冒頭の `#### 6-7: 月別サマリーHTML再生成` から、以下のような疑似コードを含む JS ブロックまで：

```javascript
// ★ ここから月別サマリー.html と フィルター付き.html を生成する
// 月は昇順ソート（a.localeCompare(b)）、未分類は末尾
// 金額フォーマット: null/undefined → '未定', 数値 → '¥' + toLocaleString('ja-JP')
// フィルター付きは var D=JSON.stringify(entries) でデータ埋め込み、render()関数で描画
// 生成後、PUT APIで上書き:
//   /_api/web/getfilebyserverrelativeurl('{encodedPath}')/$value
//   method: PUT, Content-Type: text/html; charset=utf-8

return 'summary HTMLs regenerated';
```

---

## 置換後（新規・これに差し替え）

以下のセクション全体を、SKILL.md の該当部分にそのまま貼り付けてください。

---

#### 6-7: 月別サマリーHTML再生成

案件データ更新後、ドキュメントライブラリの全案件HTMLメタデータを読み込んで月別サマリーHTMLを再生成する。
**2ファイルとも毎回上書き更新する。** パスは固定:
- `案件管理/月別サマリー.html` — 静的HTML。月ごとにテーブル表示。**月は日付昇順（古い順→新しい順）**、未分類は末尾。各月に小計行を付ける。
- `案件管理/月別サマリー_フィルター付き.html` — 全データをJSON配列として埋め込み、JavaScriptでフィルター・ソート・集計する自己完結型HTML。ローカルDLでも動作する。

> ⚠️ **【重要】スキーマ契約（Anken Sales Dashboard artifact と整合）**
>
> `月別サマリー.html` の表スキーマは Cowork artifact **「Anken Sales Dashboard」** と連携している。列の増減・順序変更・表示形式変更は Dashboard を破壊する。**両方を同時に更新すること。**
>
> **固定11列スキーマ（contract v2 / 2026-04-23〜）:**
>
> | # | 列名 | 内容 | 形式 |
> |---|---|---|---|
> | 0 | 案件ID | `Title` 列相当 | `YYYYMMDD-NN` 文字列 |
> | 1 | 案件名 | 案件名（ファイル名でなく公演名） | テキスト |
> | 2 | クライアント | `OData__x30af__x30e9__x30a4__x30a2__x30f3__x30c8_` | テキスト |
> | 3 | ステータス | `OData__x30b9__x30c6__x30fc__x30bf__x30b9_0` | `<span class="status" style="background:#xxx">...</span>` 可 |
> | 4 | 公演日 | `OData__x516c__x6f14__x65e5_` を変換 | **`YYYY/MM/DD`（スラッシュ区切り必須）** |
> | 5 | 単価(税抜) | `OData__x5358__x4fa1___x7a0e__x629c_` | `¥XX,XXX` または `未定` |
> | 6 | 室数 | `OData__x5ba4__x6570_` | 整数または空 |
> | 7 | 泊数 | `OData__x6cca__x6570_` | 整数または空 |
> | 8 | 売上(税抜) | `OData__x58f2__x4e0a___x7a0e__x629c_` | `¥XXX,XXX` または `未定` |
> | 9 | キャンセル料発生 | `OData__x30ad__x30e3__x30f3__x30bb__x30eb__x6599__x767a__x751f__x65e5_` | 日付文字列 |
> | 10 | 詳細 | 案件詳細HTMLへのリンク | `<a href="...案件管理/YYYYMM/xxx_案件詳細.html">詳細</a>` |
>
> **小計行は `<tr style="background:#e2e6ea;font-weight:bold;">` で示す**（Dashboardはこのinline styleを見てスキップする）
>
> **簡易版（会場/担当/メモ/ホテル数列の混入）や列削減は絶対に行わない。** 「メモ」「会場」等をどうしても残したい場合は **列10の右側に追加**すること（Dashboardは tds[0]〜tds[10] しか見ないため無害）。

**再生成の手順:**
1. ドキュメントライブラリの全HTMLファイル（`_案件詳細.html` を含む）のメタデータを`$select`で取得（FileRef, FileDirRef, 各カラム）
2. FSObjType===0 かつ FileLeafRef が `案件詳細.html` で終わるもののみ抽出
3. FileDirRef の末尾フォルダ名（YYYYMM/未分類）でグループ化

**以下のコードブロックをそのまま実行すること（疑似コードではなく完成コード。改変・省略禁止）:**

~~~javascript
// ★ 月別サマリー.html と 月別サマリー_フィルター付き.html を同時生成するドロップイン実装
// これは anken-sales-dashboard artifact と契約された 11列スキーマ固定。改変禁止。
(async () => {
  const siteUrl = 'https://bigholiday.sharepoint.com/sites/msteams_d9467a';
  const digest = window._spDigest;
  const libTitle = 'ドキュメント';
  const fields = [
    'Id','FileLeafRef','FileRef','FileDirRef','FSObjType',
    'OData__x30af__x30e9__x30a4__x30a2__x30f3__x30c8_',              // クライアント
    'OData__x30b9__x30c6__x30fc__x30bf__x30b9_0',                     // ステータス
    'OData__x516c__x6f14__x65e5_',                                    // 公演日
    'OData__x5358__x4fa1___x7a0e__x629c_',                            // 単価_税抜
    'OData__x5ba4__x6570_',                                           // 室数
    'OData__x6cca__x6570_',                                           // 泊数
    'OData__x58f2__x4e0a___x7a0e__x629c_',                            // 売上_税抜
    'OData__x30ad__x30e3__x30f3__x30bb__x30eb__x6599__x767a__x751f__x65e5_', // キャンセル料発生日
    'OData__x6848__x4ef6__x540d_'                                     // 案件名（文字列フィールド）
  ].join(',');

  // --- 1. 全案件HTMLメタデータを取得（ページング対応）---
  let allItems = [];
  let nextUrl = siteUrl + "/_api/web/lists/getbytitle('" + encodeURIComponent(libTitle) + "')/items?$select=" + fields + '&$top=200';
  while (nextUrl) {
    const resp = await fetch(nextUrl, { headers: { 'Accept': 'application/json;odata=verbose' } });
    const data = await resp.json();
    allItems = allItems.concat(data.d.results);
    nextUrl = data.d.__next || null;
  }
  const htmlFiles = allItems.filter(i => i.FSObjType === 0 && i.FileLeafRef.endsWith('.html') && i.FileLeafRef.includes('案件詳細'));

  // --- 2. 整形ヘルパー ---
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const fmtMoney = n => (n == null || n === '' || isNaN(Number(n))) ? '未定' : '¥' + Number(n).toLocaleString('ja-JP');
  const fmtDate = s => {  // ISO → YYYY/MM/DD（契約形式）
    if (!s) return '';
    const m = String(s).match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
    return m ? `${m[1]}/${String(m[2]).padStart(2,'0')}/${String(m[3]).padStart(2,'0')}` : s;
  };
  const statusColor = s => {
    s = s || '';
    if (/キャンセル/.test(s)) return '#d32f2f';
    if (/本予約確定|精算完了|完了|報告済/.test(s) && !/キャンセル/.test(s)) return '#2e7d32';
    if (/仮予約/.test(s)) return '#f57c00';
    if (/依頼中|対応中|調整中/.test(s)) return '#0288d1';
    return '#555';
  };
  const isConfirmed = s => /本予約確定|精算完了|完了|報告済|精算中/.test(s || '') && !/キャンセル/.test(s || '');
  const extractCaseId = leaf => { const m = leaf.match(/^(\d{8}(?:-\d+)?)_/); return m ? m[1] : ''; };
  const stripToName = leaf => leaf.replace(/^\d{8}(?:-\d+)?_/, '').replace(/_案件詳細\.html$/, '');
  const folderOf = dirRef => { const m = dirRef.match(/案件管理\/([^/]+)$/); return m ? m[1] : '未分類'; };

  // --- 3. レコード化 & 月ごとグループ化 ---
  const entries = htmlFiles.map(i => ({
    month: /^\d{6}$/.test(folderOf(i.FileDirRef)) ? folderOf(i.FileDirRef) : '未分類',
    caseId: extractCaseId(i.FileLeafRef),
    caseName: i.OData__x6848__x4ef6__x540d_ || stripToName(i.FileLeafRef),
    client: i.OData__x30af__x30e9__x30a4__x30a2__x30f3__x30c8_ || '',
    status: i.OData__x30b9__x30c6__x30fc__x30bf__x30b9_0 || '',
    kouenbi: fmtDate(i.OData__x516c__x6f14__x65e5_),
    tanka: i.OData__x5358__x4fa1___x7a0e__x629c_,
    shitsu: i.OData__x5ba4__x6570_,
    haku: i.OData__x6cca__x6570_,
    uriage: i.OData__x58f2__x4e0a___x7a0e__x629c_,
    cancelDate: i.OData__x30ad__x30e3__x30f3__x30bb__x30eb__x6599__x767a__x751f__x65e5_ || '',
    fileRef: i.FileRef
  })).sort((a,b) => (a.kouenbi||'').localeCompare(b.kouenbi||''));

  const byMonth = {};
  entries.forEach(e => { (byMonth[e.month] = byMonth[e.month] || []).push(e); });
  const months = Object.keys(byMonth).sort((a,b) => a === '未分類' ? 1 : b === '未分類' ? -1 : a.localeCompare(b));

  // --- 4. 月別サマリー.html （静的11列 + 月別小計） ---
  const rowHtml = r => {
    const href = 'https://bigholiday.sharepoint.com' + encodeURI(r.fileRef);
    return `<tr>
      <td>${esc(r.caseId)}</td>
      <td>${esc(r.caseName)}</td>
      <td>${esc(r.client)}</td>
      <td><span class="status" style="background:${statusColor(r.status)}">${esc(r.status)}</span></td>
      <td>${esc(r.kouenbi)}</td>
      <td class="num">${fmtMoney(r.tanka)}</td>
      <td class="num">${r.shitsu == null || r.shitsu === '' ? '' : esc(r.shitsu)}</td>
      <td class="num">${r.haku == null || r.haku === '' ? '' : esc(r.haku)}</td>
      <td class="num">${fmtMoney(r.uriage)}</td>
      <td>${esc(r.cancelDate)}</td>
      <td><a href="${esc(href)}" target="_blank">詳細</a></td>
    </tr>`;
  };
  const monthBlocks = months.map(m => {
    const rows = byMonth[m];
    const subtotal = rows.filter(r => isConfirmed(r.status) && r.uriage != null && r.uriage !== '').reduce((s,r) => s + Number(r.uriage), 0);
    const label = m === '未分類' ? '未分類' : `${m.slice(0,4)}年${m.slice(4)}月`;
    return `<h2 id="m${m}">${label}（${rows.length}件）</h2>
<table><thead><tr><th>案件ID</th><th>案件名</th><th>クライアント</th><th>ステータス</th><th>公演日</th><th class="num">単価(税抜)</th><th class="num">室数</th><th class="num">泊数</th><th class="num">売上(税抜)</th><th>キャンセル料発生</th><th>詳細</th></tr></thead>
<tbody>${rows.map(rowHtml).join('')}
<tr style="background:#e2e6ea;font-weight:bold;"><td colspan="8" style="text-align:right;">確定売上 小計</td><td class="num">${fmtMoney(subtotal)}</td><td colspan="2"></td></tr>
</tbody></table>`;
  }).join('');
  const generatedAt = new Date().toISOString().slice(0,19).replace('T',' ');
  const navLinks = months.map(m => `<a href="#m${m}">${m}</a>`).join('');
  const summaryHtml = `<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8"><title>月別サマリー</title>
<style>body{font-family:'Yu Gothic',YuGothic,'Meiryo',sans-serif;max-width:1500px;margin:20px auto;padding:0 20px;color:#333;}
h1{border-bottom:3px solid #0078d4;padding-bottom:8px;color:#0078d4;}
h2{background:#f3f6fb;padding:10px 14px;border-left:6px solid #0078d4;margin-top:30px;font-size:18px;}
table{width:100%;border-collapse:collapse;margin:10px 0 20px;background:#fff;font-size:13px;}
th{background:#0078d4;color:#fff;padding:8px;text-align:left;border:1px solid #005a9e;}
td{padding:6px 8px;border:1px solid #ddd;vertical-align:top;}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;}
tr:nth-child(even){background:#f9fbfd;}
.status{font-weight:bold;padding:2px 6px;border-radius:3px;color:#fff;display:inline-block;font-size:11px;}
.meta{color:#777;font-size:11px;}
a{color:#0078d4;text-decoration:none;}
a:hover{text-decoration:underline;}
.nav{background:#f0f4f8;padding:10px;border-radius:4px;margin:10px 0;}
.nav a{margin-right:15px;}</style>
</head><body>
<h1>案件一覧 月別サマリー</h1>
<p class="meta">生成日時: ${generatedAt} / 対象案件数: ${entries.length}件 / schema: contract-v2-11col</p>
<div class="nav">${navLinks}</div>
${monthBlocks}</body></html>`;

  // --- 5. 月別サマリー_フィルター付き.html （11列 + JSONデータ埋め込み + JSフィルター） ---
  const dataForEmbed = entries.map(r => ({
    month: r.month,
    caseId: r.caseId,
    caseName: r.caseName,
    client: r.client,
    status: r.status,
    kouenbi: r.kouenbi,
    tanka: r.tanka,
    shitsu: r.shitsu,
    haku: r.haku,
    uriage: r.uriage,
    cancelDate: r.cancelDate,
    detailUrl: 'https://bigholiday.sharepoint.com' + encodeURI(r.fileRef)
  }));
  const dataJson = JSON.stringify(dataForEmbed).replace(/</g, '\\u003c').replace(/-->/g, '--\\u003e');
  const filterHtml = `<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8"><title>月別サマリー（フィルター付き）</title>
<style>body{font-family:'Yu Gothic',YuGothic,Meiryo,sans-serif;max-width:1600px;margin:20px auto;padding:0 20px;color:#333;}
h1{border-bottom:3px solid #0078d4;padding-bottom:8px;color:#0078d4;}
.filters{background:#f3f6fb;padding:14px;border-radius:6px;margin:10px 0 20px;display:flex;flex-wrap:wrap;gap:15px;align-items:center;}
.filters label{display:flex;flex-direction:column;gap:4px;font-size:12px;color:#555;}
.filters select,.filters input{padding:6px 8px;border:1px solid #bbb;border-radius:4px;font-size:13px;min-width:120px;}
.filters button{padding:8px 14px;background:#0078d4;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:13px;}
table{width:100%;border-collapse:collapse;font-size:12px;background:#fff;}
th{background:#0078d4;color:#fff;padding:8px;text-align:left;border:1px solid #005a9e;cursor:pointer;user-select:none;}
th:hover{background:#005a9e;}
td{padding:6px 8px;border:1px solid #ddd;vertical-align:top;}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;}
tr:nth-child(even){background:#f9fbfd;}
.status{font-weight:bold;padding:2px 6px;border-radius:3px;color:#fff;display:inline-block;font-size:11px;}
.meta{color:#777;font-size:12px;}
.summary-bar{background:#eaf4fb;border:1px solid #bfdbfe;padding:10px 14px;margin:10px 0;border-radius:4px;font-size:13px;}
a{color:#0078d4;text-decoration:none;}a:hover{text-decoration:underline;}</style>
</head><body>
<h1>案件一覧 月別サマリー（フィルター付き）</h1>
<p class="meta">生成日時: ${generatedAt} / 全${entries.length}件 / schema: contract-v2-11col</p>
<div class="filters">
<label>計上月<select id="fMonth"><option value="">全て</option>${months.map(m => `<option value="${m}">${m}</option>`).join('')}</select></label>
<label>クライアント<select id="fClient"><option value="">全て</option></select></label>
<label>ステータス<select id="fStatus"><option value="">全て</option></select></label>
<label>確定フィルタ<select id="fConf"><option value="">すべて</option><option value="confirmed">確定のみ</option><option value="excludeCancel">キャンセル除く</option></select></label>
<label>検索<input type="text" id="fSearch" placeholder="案件名/クライアント"></label>
<button onclick="resetFilter()">リセット</button>
</div>
<div class="summary-bar" id="summaryBar"></div>
<table id="mainTable"><thead><tr>
<th data-k="month">計上月</th><th data-k="caseId">案件ID</th><th data-k="caseName">案件名</th><th data-k="client">クライアント</th><th data-k="status">ステータス</th><th data-k="kouenbi">公演日</th><th data-k="tanka" class="num">単価(税抜)</th><th data-k="shitsu" class="num">室数</th><th data-k="haku" class="num">泊数</th><th data-k="uriage" class="num">売上(税抜)</th><th data-k="cancelDate">キャンセル料発生</th><th>詳細</th>
</tr></thead><tbody id="tbody"></tbody></table>
<script>
const D=${dataJson};
const statusColor=s=>{s=s||'';if(/キャンセル/.test(s))return'#d32f2f';if(/本予約確定|精算完了|完了|報告済/.test(s)&&!/キャンセル/.test(s))return'#2e7d32';if(/仮予約/.test(s))return'#f57c00';if(/依頼中|対応中|調整中/.test(s))return'#0288d1';return'#555';};
const isConfirmed=s=>/本予約確定|精算完了|完了|報告済|精算中/.test(s||'')&&!/キャンセル/.test(s||'');
const isCancel=s=>/キャンセル/.test(s||'');
const fmt=n=>(n==null||n===''||isNaN(Number(n)))?'未定':'¥'+Number(n).toLocaleString('ja-JP');
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let sortKey=null,sortAsc=true;
function populateOptions(){
 const cs=Array.from(new Set(D.map(d=>d.client).filter(Boolean))).sort();
 const ss=Array.from(new Set(D.map(d=>d.status).filter(Boolean))).sort();
 document.getElementById('fClient').innerHTML='<option value="">全て</option>'+cs.map(c=>'<option>'+esc(c)+'</option>').join('');
 document.getElementById('fStatus').innerHTML='<option value="">全て</option>'+ss.map(c=>'<option>'+esc(c)+'</option>').join('');
}
function apply(){
 const fM=document.getElementById('fMonth').value,fC=document.getElementById('fClient').value,fS=document.getElementById('fStatus').value,fConf=document.getElementById('fConf').value,fQ=document.getElementById('fSearch').value.toLowerCase();
 let rows=D.filter(d=>{
  if(fM&&d.month!==fM)return false;
  if(fC&&d.client!==fC)return false;
  if(fS&&d.status!==fS)return false;
  if(fConf==='confirmed'&&!isConfirmed(d.status))return false;
  if(fConf==='excludeCancel'&&isCancel(d.status))return false;
  if(fQ){const h=(d.caseName+' '+d.client+' '+d.status).toLowerCase();if(!h.includes(fQ))return false;}
  return true;
 });
 if(sortKey){rows.sort((a,b)=>{const va=a[sortKey]??'',vb=b[sortKey]??'';if(typeof va==='number'&&typeof vb==='number')return sortAsc?va-vb:vb-va;return sortAsc?String(va).localeCompare(String(vb)):String(vb).localeCompare(String(va));});}
 document.getElementById('tbody').innerHTML=rows.map(d=>\`<tr>
  <td>\${esc(d.month)}</td><td>\${esc(d.caseId)}</td><td>\${esc(d.caseName)}</td><td>\${esc(d.client)}</td>
  <td><span class="status" style="background:\${statusColor(d.status)}">\${esc(d.status)}</span></td>
  <td>\${esc(d.kouenbi)}</td><td class="num">\${fmt(d.tanka)}</td>
  <td class="num">\${d.shitsu==null||d.shitsu===''?'':esc(d.shitsu)}</td>
  <td class="num">\${d.haku==null||d.haku===''?'':esc(d.haku)}</td>
  <td class="num">\${fmt(d.uriage)}</td><td>\${esc(d.cancelDate)}</td>
  <td><a href="\${esc(d.detailUrl)}" target="_blank">詳細</a></td></tr>\`).join('');
 const tot=rows.reduce((s,r)=>s+Number(r.uriage||0),0);
 const cfm=rows.filter(r=>isConfirmed(r.status)).reduce((s,r)=>s+Number(r.uriage||0),0);
 document.getElementById('summaryBar').innerHTML=\`表示: <strong>\${rows.length}件</strong> / 全\${D.length}件 ｜ 売上合計: <strong>\${fmt(tot)}</strong> ｜ 確定売上: <strong>\${fmt(cfm)}</strong>\`;
}
function resetFilter(){['fMonth','fClient','fStatus','fConf','fSearch'].forEach(id=>document.getElementById(id).value='');sortKey=null;sortAsc=true;apply();}
document.querySelectorAll('th[data-k]').forEach(th=>th.addEventListener('click',()=>{const k=th.getAttribute('data-k');if(sortKey===k)sortAsc=!sortAsc;else{sortKey=k;sortAsc=true;}apply();}));
['fMonth','fClient','fStatus','fConf'].forEach(id=>document.getElementById(id).addEventListener('change',apply));
document.getElementById('fSearch').addEventListener('input',apply);
populateOptions();apply();
</script></body></html>`;

  // --- 6. PUTで上書きアップロード ---
  const putFile = async (serverRelUrl, body) => {
    return fetch(siteUrl + "/_api/web/getfilebyserverrelativeurl('" + encodeURIComponent(serverRelUrl) + "')/$value", {
      method: 'POST',
      headers: {
        'Accept': 'application/json;odata=verbose',
        'Content-Type': 'text/html; charset=utf-8',
        'X-RequestDigest': digest,
        'X-HTTP-Method': 'PUT'
      },
      body: new TextEncoder().encode(body)
    });
  };
  const r1 = await putFile('/sites/msteams_d9467a/Shared Documents/案件管理/月別サマリー.html', summaryHtml);
  const r2 = await putFile('/sites/msteams_d9467a/Shared Documents/案件管理/月別サマリー_フィルター付き.html', filterHtml);

  return JSON.stringify({
    status: 'OK',
    contract: 'v2-11col-2026-04-23',
    entries: entries.length,
    months: months.length,
    summaryStatus: r1.status,
    filterStatus: r2.status
  });
})()
~~~

**実装上の不変条件（アサートすべきポイント）:**
- 生成する `<thead><tr>` は**必ず11個の `<th>` を含む** — 減らしてはならない
- 公演日セル（tds[4]）は**必ず `YYYY/MM/DD` 形式**（ISO `YYYY-MM-DD` はNG）
- 小計行は `style="background:#e2e6ea;..."` を含むこと（Dashboardのスキップ判定で使用）
- 詳細列のリンクは `案件管理/YYYYMM/...案件詳細.html` パターンを含むこと（Dashboardのフォルダ抽出regexで使用）
