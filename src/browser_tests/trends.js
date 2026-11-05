exports.run = async (p, w) => {
  await p.waitFor("document.querySelectorAll('#tr-line path').length >= 1");
  p.log("series drawn:", await p.eval("document.querySelectorAll('#tr-line path').length"), "|", await p.eval("document.getElementById('tr-summary').textContent"));
  await p.click('.chip[data-i="4"]'); await p.click('.chip[data-i="7"]');
  p.log("after toggles:", await p.eval("document.querySelectorAll('#tr-line path').length"));
  await p.eval("(()=>{const s=document.getElementById('tr-make');s.value=String([...s.options].findIndex(o=>o.text==='FORD')-1);s.dispatchEvent(new Event('change'))})()");
  p.log("Ford:", await p.eval("document.getElementById('tr-summary').textContent"));
  await p.eval("(()=>{const s=document.getElementById('tr-year');s.value=String([...s.options].findIndex(o=>o.text==='2020')-1);s.dispatchEvent(new Event('change'))})()");
  p.log("Ford 2020:", await p.eval("document.getElementById('tr-summary').textContent"));
  await p.eval("(()=>{const s=document.getElementById('tr-make');s.value='-1';s.dispatchEvent(new Event('change'));const y=document.getElementById('tr-year');y.value='-1';y.dispatchEvent(new Event('change'))})()");
  p.log("reset:", await p.eval("document.getElementById('tr-summary').textContent"));
  await p.eval(`(()=>{const r=document.getElementById('tr-line').getBoundingClientRect();document.getElementById('tr-line').dispatchEvent(new PointerEvent('pointermove',{clientX:r.left+r.width*0.6,clientY:r.top+50,bubbles:true}))})()`);
  p.log("tooltip:", await p.eval("document.getElementById('tr-tip').hidden + ' ' + document.getElementById('tr-tip').textContent.slice(0,90)"));
  await p.eval("document.getElementById('tr-line').dispatchEvent(new PointerEvent('pointerleave'))");
  await p.shot("explore_trends");
};
