exports.run = async (p, w) => {
  await p.waitFor("!document.getElementById('try-go').disabled", 60000);
  p.log("model loaded:", await p.eval("document.getElementById('try-load-text').textContent"));
  await p.type("#try-text", "The steering wheel locked up while I was driving on the highway and I could not turn the car.");
  await p.click("#try-go");
  await p.waitFor("document.querySelectorAll('#try-out .bar-row').length === 3");
  p.log("top-3:", await p.eval("[...document.querySelectorAll('#try-out .bar-row')].map(e=>e.textContent).join(' | ')"));
  p.log("highlighted words:", await p.eval("document.querySelectorAll('#try-out .wi-text .sup, #try-out .wi-text .arg').length"));
  await p.click("#try-real");
  await p.waitFor("document.querySelector('#try-out .verdict')");
  p.log("verdict:", await p.eval("document.querySelector('#try-out .verdict').textContent"));
  await p.shot("explore_try");
};
