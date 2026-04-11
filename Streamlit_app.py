<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GST Reconciliation Dashboard</title>

<!-- XLSX Library -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>

<style>
body {
  font-family: Arial;
  background: #f4f7fb;
  margin: 0;
}

/* HEADER */
.header {
  background: linear-gradient(90deg, #1f4e79, #4fa3d1);
  color: white;
  padding: 25px;
  text-align: center;
}

/* UPLOAD */
.upload-container {
  display: flex;
  justify-content: space-around;
  margin: 30px;
  flex-wrap: wrap;
}

.upload-box {
  background: white;
  padding: 25px;
  border-radius: 12px;
  border: 2px dashed #ccc;
  text-align: center;
  width: 40%;
  min-width: 280px;
  transition: 0.3s;
}

.upload-box:hover {
  border-color: #1f77b4;
  background: #f0f8ff;
}

/* BUTTON */
.center {
  text-align: center;
  margin: 20px;
}

button {
  padding: 12px 25px;
  border: none;
  border-radius: 10px;
  background: linear-gradient(90deg, #1f77b4, #4fa3d1);
  color: white;
  font-weight: bold;
  cursor: pointer;
  transition: 0.3s;
}

button:hover {
  transform: scale(1.05);
}

/* PROGRESS */
.progress-container {
  width: 80%;
  margin: auto;
  background: #ddd;
  border-radius: 10px;
  overflow: hidden;
}

#progressBar {
  width: 0%;
  height: 15px;
  background: linear-gradient(90deg, #28a745, #5cd65c);
  transition: width 0.3s;
}

/* CARDS */
.cards {
  display: flex;
  justify-content: space-around;
  margin: 30px;
  flex-wrap: wrap;
}

.card {
  background: white;
  padding: 20px;
  border-radius: 12px;
  width: 25%;
  min-width: 250px;
  text-align: center;
  box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
  margin: 10px;
}

/* TABLE */
.table-container {
  margin: 30px;
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  background: white;
}

th, td {
  padding: 10px;
  border: 1px solid #ddd;
}

th {
  background: #1f4e79;
  color: white;
}

.match {
  background-color: #e6ffed;
}

.unmatch {
  background-color: #ffe6e6;
}

footer {
  text-align: center;
  color: gray;
  margin: 20px;
}
</style>
</head>

<body>

<!-- HEADER -->
<div class="header">
  <h1>📘 GST Reconciliation Dashboard</h1>
  <p>Smart comparison of GSTR-2B & Purchase Register</p>
</div>

<!-- UPLOAD -->
<div class="upload-container">
  <div class="upload-box">
    <h3>📄 Upload GSTR-2B</h3>
    <input type="file" id="gstFile">
  </div>

  <div class="upload-box">
    <h3>📄 Upload Purchase Register</h3>
    <input type="file" id="purchaseFile">
  </div>
</div>

<!-- BUTTON -->
<div class="center">
  <button onclick="runReco()">🚀 Run Reconciliation</button>
</div>

<!-- PROGRESS -->
<div class="progress-container">
  <div id="progressBar"></div>
</div>

<!-- SUMMARY -->
<div class="cards">
  <div class="card">
    <h4>📄 Total Records</h4>
    <h2 id="total">0</h2>
  </div>

  <div class="card">
    <h4>✅ Matched</h4>
    <h2 id="matched">0</h2>
    <p id="pct"></p>
  </div>

  <div class="card">
    <h4>❌ Unmatched</h4>
    <h2 id="unmatched">0</h2>
  </div>
</div>

<!-- TABLE -->
<div class="table-container">
  <h3>📋 Detailed Results</h3>
  <table id="table">
    <thead></thead>
    <tbody></tbody>
  </table>
</div>

<!-- DOWNLOAD -->
<div class="center">
  <button onclick="downloadExcel()">⬇️ Download Excel</button>
</div>

<footer>
GST Reco Pro • Built with HTML + JS 🚀
</footer>

<script>
let resultData = [];

// READ EXCEL
function readExcel(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const data = new Uint8Array(e.target.result);
      const workbook = XLSX.read(data, { type: "array" });
      const sheet = workbook.Sheets[workbook.SheetNames[0]];
      resolve(XLSX.utils.sheet_to_json(sheet));
    };
    reader.readAsArrayBuffer(file);
  });
}

// RUN RECO
async function runReco() {
  const gstFile = document.getElementById("gstFile").files[0];
  const purchaseFile = document.getElementById("purchaseFile").files[0];

  if (!gstFile || !purchaseFile) {
    alert("Please upload both files ❗");
    return;
  }

  // Progress animation
  let progress = 0;
  const bar = document.getElementById("progressBar");

  const interval = setInterval(() => {
    progress += 20;
    bar.style.width = progress + "%";
    if (progress >= 100) clearInterval(interval);
  }, 200);

  const gst = await readExcel(gstFile);
  const books = await readExcel(purchaseFile);

  resultData = processReco(gst, books);

  renderSummary();
  renderTable();
}

// SIMPLE MATCH LOGIC
function processReco(gst, books) {
  return gst.map(g => {
    const match = books.find(b => b.Invoice === g.Invoice);
    return {
      ...g,
      Match_Status: match ? "Matched" : "Unmatched"
    };
  });
}

// SUMMARY
function renderSummary() {
  const total = resultData.length;
  const matched = resultData.filter(d => d.Match_Status === "Matched").length;
  const unmatched = total - matched;
  const pct = total ? ((matched / total) * 100).toFixed(1) : 0;

  document.getElementById("total").innerText = total;
  document.getElementById("matched").innerText = matched;
  document.getElementById("unmatched").innerText = unmatched;
  document.getElementById("pct").innerText = pct + "%";
}

// TABLE
function renderTable() {
  const tableHead = document.querySelector("#table thead");
  const tableBody = document.querySelector("#table tbody");

  tableHead.innerHTML = "";
  tableBody.innerHTML = "";

  if (resultData.length === 0) return;

  const headers = Object.keys(resultData[0]);

  let headRow = "<tr>";
  headers.forEach(h => headRow += `<th>${h}</th>`);
  headRow += "</tr>";
  tableHead.innerHTML = headRow;

  resultData.forEach(row => {
    let tr = `<tr class="${row.Match_Status === 'Matched' ? 'match' : 'unmatch'}">`;
    headers.forEach(h => tr += `<td>${row[h]}</td>`);
    tr += "</tr>";
    tableBody.innerHTML += tr;
  });
}

// DOWNLOAD
function downloadExcel() {
  if (resultData.length === 0) {
    alert("No data to download ❗");
    return;
  }

  const ws = XLSX.utils.json_to_sheet(resultData);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Reco");

  XLSX.writeFile(wb, "GST_Reconciliation.xlsx");
}
</script>

</body>
</html>
