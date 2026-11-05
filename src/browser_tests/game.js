exports.run = async (p, w) => {
  await p.waitFor("document.querySelectorAll('.choice').length === 10");
  for (let r = 0; r < 10; r++) {
    await p.waitFor("document.querySelectorAll('.choice').length === 10 && !document.querySelector('.choice').disabled");
    if (r === 0) await p.shot("explore_game_round");
    await p.click(".choice:nth-child(" + (1 + (r % 10)) + ")");
    await p.waitFor("document.getElementById('g-next')");
    if (r === 1) await p.shot("explore_game_reveal");
    p.log("round", r + 1, "live:", await p.eval("document.getElementById('g-live').textContent"));
    await p.click("#g-next");
  }
  await p.waitFor("document.getElementById('g-again')");
  p.log("final:", await p.eval("document.querySelector('.final h2').textContent + ' / ' + document.querySelector('.final p').textContent"));
  await p.shot("explore_game_final");
  await p.click("#g-again");
  await p.waitFor("document.getElementById('g-round').textContent === 'Round 1 of 10'");
  p.log("play again works");
};
