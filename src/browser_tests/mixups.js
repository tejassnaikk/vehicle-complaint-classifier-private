exports.run = async (p, w) => {
  await p.waitFor("document.querySelectorAll('.mx-cell').length === 100");
  p.log("fallback hidden:", await p.eval("document.getElementById('mx-fallback').hidden"));
  // biggest off-diagonal cell
  const cell = await p.eval(`(()=>{let b=null;document.querySelectorAll('.mx-cell').forEach(e=>{if(e.dataset.i!==e.dataset.j&&(!b||+e.textContent>+b.textContent))b=e});return b.dataset.i+','+b.dataset.j+':'+b.textContent})()`);
  p.log("largest off-diagonal cell (i,j:count):", cell);
  const [ij] = cell.split(":");
  await p.click(`.mx-cell[data-i="${ij.split(",")[0]}"][data-j="${ij.split(",")[1]}"]`);
  await p.waitFor("document.querySelectorAll('#mx-detail .mx-card').length > 0");
  p.log("cards:", await p.eval("document.querySelectorAll('#mx-detail .mx-card').length"), await p.eval("document.querySelector('#mx-detail h3').textContent"));
  await p.click("#mx-hide");
  p.log("diag hidden cells:", await p.eval("document.querySelectorAll('.hidden-diag').length"));
  await p.click("#mx-hide");
  await p.shot("explore_mixups");
};
