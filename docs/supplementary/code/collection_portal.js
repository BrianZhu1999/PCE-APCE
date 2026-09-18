(() => {
  'use strict';
  const movies = JSON.parse(document.getElementById('movie-data').textContent);
  const cards = [...document.querySelectorAll('.movie-card')];
  const previews = [...document.querySelectorAll('.inline-video')];
  const dialog = document.getElementById('video-dialog');
  const player = document.getElementById('dialog-video');
  const closeButton = document.getElementById('close-player');
  const positions = new Map();
  const selectedVariants = new Map();
  const previewGenerations = new WeakMap();
  let language = 'en';
  let active = -1;
  let origin = null;
  let scrollBeforeOpen = 0;
  let generation = 0;
  try { language = localStorage.getItem('supplementary-portal-language') === 'zh' ? 'zh' : 'en'; } catch (_) {}
  const text = (zh,en) => language === 'zh' ? zh : en;
  const movieTitle = movie => language === 'zh' ? movie.zh : movie.en;
  const label = n => text(`补充视频 ${n}`,`Supplementary Movie ${n}`);
  const variantFor = index => movies[index].variants?.find(v=>v.id===selectedVariants.get(index)) || movies[index].variants?.[0] || movies[index];
  const mediaURL = variant => variant.file + (variant.revision ? `?v=${encodeURIComponent(variant.revision)}` : '');
  const posterURL = variant => variant.poster + (variant.revision ? `?v=${encodeURIComponent(variant.revision)}` : '');
  function variantControls(index, location) {
    const variants=movies[index].variants;
    if(!variants)return null;
    const wrapper=document.createElement('div');wrapper.className='movie-variants';wrapper.dataset.variantMovie=String(index);
    const choices=document.createElement('div');choices.className='variant-choices';choices.setAttribute('role','group');choices.dataset.ariaZh='三维声场显示方式';choices.dataset.ariaEn='3D field visualization';
    for(const v of variants){const button=document.createElement('button');button.type='button';button.dataset.variant=v.id;button.dataset.zh=v.zh;button.dataset.en=v.en;button.addEventListener('click',()=>switchVariant(index,v.id,location));choices.append(button)}
    wrapper.append(choices);
    if(location==='card'){const downloads=document.createElement('div');downloads.className='variant-downloads';for(const v of variants){const link=document.createElement('a');link.href=mediaURL(v);link.download=v.file;link.dataset.zh=`下载${v.zh}版`;link.dataset.en=`Download ${v.en.toLowerCase()}`;downloads.append(link)}wrapper.append(downloads)}
    return wrapper;
  }
  function updateVariants(){document.querySelectorAll('[data-variant-movie]').forEach(group=>{const index=Number(group.dataset.variantMovie),chosen=variantFor(index).id;group.querySelectorAll('[data-variant]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.variant===chosen)))});}
  function switchVariant(index,id,location){
    if(variantFor(index).id===id)return;
    const inDialog=active===index&&dialog.open,source=inDialog?player:previews[index];
    const resume=source.readyState>0&&Number.isFinite(source.currentTime)?source.currentTime:(positions.get(index)||0),playing=!source.paused;
    positions.set(index,resume);selectedVariants.set(index,id);
    const video=previews[index],variant=variantFor(index),token=(previewGenerations.get(video)||0)+1;
    previewGenerations.set(video,token);video.pause();video.src=mediaURL(variant);video.poster=posterURL(variant);
    video.addEventListener('loadedmetadata',()=>{if(previewGenerations.get(video)!==token)return;video.currentTime=Math.min(resume,Math.max(0,video.duration-.01));if(playing&&!inDialog)video.play().catch(()=>{});},{once:true});video.load();
    if(inDialog){player.pause();loadMovie(index,playing)}
    updateVariants();
  }
  function applyLanguage() {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-zh][data-en]').forEach(el => {el.textContent = el.dataset[language];});
    document.querySelectorAll('[data-aria-zh]').forEach(el => {el.setAttribute('aria-label',el.dataset[language === 'zh' ? 'ariaZh' : 'ariaEn']);});
    document.querySelectorAll('[data-language]').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.language === language)));
    document.title = text('补充视频与交互材料','Supplementary Movies');
    previews.forEach((v,i)=>v.setAttribute('aria-label',label(movies[i].number)+': '+movieTitle(movies[i])));
    previews.forEach((v,i)=>updatePreviewButton(i));
    if (active >= 0) updatePlayerTitle();
    updateVariants();
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
    download.href = mediaURL(variantFor(active));
    download.download = variantFor(active).file;
    const existing=document.getElementById('dialog-variants');
    if(existing)existing.remove();
    const controls=variantControls(active,'dialog');
    if(controls){controls.id='dialog-variants';document.querySelector('.player-top').after(controls);controls.querySelectorAll('[data-zh][data-en]').forEach(el=>el.textContent=el.dataset[language]);controls.querySelectorAll('[data-aria-zh]').forEach(el=>el.setAttribute('aria-label',el.dataset[language==='zh'?'ariaZh':'ariaEn']));updateVariants();}
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
    video.addEventListener('timeupdate',()=>{if(video.readyState>0)positions.set(index,video.currentTime);});
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
    if (active >= 0 && player.readyState>0 && Number.isFinite(player.currentTime)) positions.set(active,player.currentTime);
  }
  function loadMovie(index, autoplay=true) {
    generation += 1;
    const token = generation;
    active = index;
    updatePlayerTitle();
    document.getElementById('player-error').hidden = true;
    player.poster = posterURL(variantFor(index));
    player.src = mediaURL(variantFor(index));
    const resume = positions.get(index) || 0;
    player.addEventListener('loadedmetadata',() => {
      if (token !== generation || !dialog.open) return;
      if (resume > 0 && resume < player.duration) player.currentTime = resume;
    },{once:true});
    player.load();
    // Native controls remain available when the browser declines autoplay.
    const pending = autoplay ? player.play() : null;
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
  movies.forEach((movie,index)=>{const controls=variantControls(index,'card');if(controls)cards[index].querySelector('.movie-description').after(controls)});
  applyLanguage();
})();
