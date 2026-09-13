(() => {
  'use strict';
  const movies = JSON.parse(document.getElementById('movie-data').textContent);
  const cards = [...document.querySelectorAll('.movie-card')];
  const previews = [...document.querySelectorAll('.inline-video')];
  const dialog = document.getElementById('video-dialog');
  const player = document.getElementById('dialog-video');
  const closeButton = document.getElementById('close-player');
  const positions = new Map();
  let language = 'en';
  let active = -1;
  let origin = null;
  let scrollBeforeOpen = 0;
  let generation = 0;
  try { language = localStorage.getItem('supplementary-portal-language') === 'zh' ? 'zh' : 'en'; } catch (_) {}
  const text = (zh,en) => language === 'zh' ? zh : en;
  const movieTitle = movie => language === 'zh' ? movie.zh : movie.en;
  const label = n => text(`补充视频 ${n}`,`Supplementary Movie ${n}`);
  function applyLanguage() {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-zh][data-en]').forEach(el => {el.textContent = el.dataset[language];});
    document.querySelectorAll('[data-aria-zh]').forEach(el => {el.setAttribute('aria-label',el.dataset[language === 'zh' ? 'ariaZh' : 'ariaEn']);});
    document.querySelectorAll('[data-language]').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.language === language)));
    document.title = text('补充视频与交互材料','Supplementary Movies');
    previews.forEach((v,i)=>v.setAttribute('aria-label',label(movies[i].number)+': '+movieTitle(movies[i])));
    previews.forEach((v,i)=>updatePreviewButton(i));
    if (active >= 0) updatePlayerTitle();
    updateFilterStatus();
  }
  function updatePlayerTitle() {
    const movie = movies[active];
    document.getElementById('player-number').textContent = label(movie.number);
    document.getElementById('player-title').textContent = movieTitle(movie);
    document.getElementById('player-counter').textContent = `${active + 1} / ${movies.length}`;
    document.getElementById('previous-movie').disabled = active === 0;
    document.getElementById('next-movie').disabled = active === movies.length - 1;
    player.setAttribute('aria-label',label(movie.number)+': '+movieTitle(movie));
    const download = document.getElementById('download-movie');
    download.href = movie.file;
    download.download = movie.file;
  }
  function pauseOthers(current) {
    [...previews,player].forEach(v => {if (v !== current) v.pause();});
  }
  function showVideoError(video,errorElement) {
    errorElement.hidden = !video.error;
  }
  function updatePreviewButton(index) {
    const button=document.querySelector(`[data-play-movie="${index}"]`);
    button.textContent=previews[index].paused?text('播放预览','Play preview'):text('暂停','Pause');
    button.setAttribute('aria-label',button.textContent+': '+movieTitle(movies[index]));
  }
  previews.forEach((video,index) => {
    video.addEventListener('play',() => {pauseOthers(video);updatePreviewButton(index);});
    video.addEventListener('pause',()=>updatePreviewButton(index));
    video.addEventListener('ended',()=>updatePreviewButton(index));
    video.addEventListener('timeupdate',()=>positions.set(index,video.currentTime));
    video.addEventListener('error',()=>showVideoError(video,cards[index].querySelector('.video-error')));
    video.addEventListener('loadeddata',()=>{cards[index].querySelector('.video-error').hidden=true;});
  });
  document.querySelectorAll('[data-play-movie]').forEach(button=>button.addEventListener('click',()=>{
    const video=previews[Number(button.dataset.playMovie)];
    if(video.paused){const pending=video.play();if(pending)pending.catch(()=>{});}else video.pause();
  }));
  player.addEventListener('play',()=>pauseOthers(player));
  player.addEventListener('error',()=>showVideoError(player,document.getElementById('player-error')));
  player.addEventListener('loadeddata',()=>{document.getElementById('player-error').hidden=true;});
  function savePosition() {
    if (active >= 0 && Number.isFinite(player.currentTime)) positions.set(active,player.currentTime);
  }
  function loadMovie(index) {
    generation += 1;
    const token = generation;
    active = index;
    updatePlayerTitle();
    document.getElementById('player-error').hidden = true;
    player.poster = movies[index].poster;
    player.src = movies[index].file;
    const resume = positions.get(index) || 0;
    player.addEventListener('loadedmetadata',() => {
      if (token !== generation || !dialog.open) return;
      if (resume > 0 && resume < player.duration) player.currentTime = resume;
    },{once:true});
    player.load();
    // Native controls remain available when the browser declines autoplay.
    const pending = player.play();
    if (pending) pending.catch(()=>{});
  }
  function openMovie(index,button) {
    origin = button;
    scrollBeforeOpen = window.scrollY;
    positions.set(index,previews[index].currentTime || positions.get(index) || 0);
    pauseOthers(null);
    document.body.classList.add('modal-open');
    dialog.showModal();
    loadMovie(index);
    closeButton.focus({preventScroll:true});
  }
  function closePlayer() { if (dialog.open) dialog.close(); }
  dialog.addEventListener('close',()=>{
    savePosition();
    generation += 1;
    player.pause();
    if (active >= 0 && previews[active].readyState > 0) {
      const resume = positions.get(active) || 0;
      try {previews[active].currentTime = resume;} catch (_) {}
    }
    player.removeAttribute('src');
    player.load();
    active = -1;
    document.body.classList.remove('modal-open');
    window.scrollTo({top:scrollBeforeOpen,behavior:'instant'});
    origin?.focus({preventScroll:true});
  });
  dialog.addEventListener('cancel',event=>{event.preventDefault();closePlayer();});
  let backdropStart = false;
  const outside = event => {
    const r = dialog.getBoundingClientRect();
    return event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom;
  };
  dialog.addEventListener('pointerdown',event=>{backdropStart=event.target===dialog && outside(event);});
  dialog.addEventListener('click',event=>{if(backdropStart && event.target===dialog && outside(event))closePlayer();backdropStart=false;});
  closeButton.addEventListener('click',closePlayer);
  document.querySelectorAll('[data-open-movie]').forEach(button=>button.addEventListener('click',()=>openMovie(Number(button.dataset.openMovie),button)));
  for (const [id,delta] of [['previous-movie',-1],['next-movie',1]]) {
    document.getElementById(id).addEventListener('click',()=>{
      const next=active+delta;
      if(next<0||next>=movies.length)return;
      savePosition();player.pause();loadMovie(next);
    });
  }
  function updateFilterStatus() {
    const count=cards.filter(c=>!c.hidden).length;
    document.getElementById('filter-status').textContent=text(`显示 ${count} 部视频`,`Showing ${count} movies`);
  }
  document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{
    const filter=button.dataset.filter;
    cards.forEach((card,i)=>{card.hidden=filter!=='all'&&card.dataset.group!==filter;if(card.hidden)previews[i].pause();});
    document.querySelectorAll('[data-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
    updateFilterStatus();
  }));
  document.querySelectorAll('[data-language]').forEach(button=>button.addEventListener('click',()=>{
    language=button.dataset.language;
    try {localStorage.setItem('supplementary-portal-language',language);} catch (_) {}
    applyLanguage();
  }));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pauseOthers(null);});
  window.addEventListener('pagehide',()=>pauseOthers(null));
  applyLanguage();
})();
