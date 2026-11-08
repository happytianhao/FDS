'use strict';
// Bounded, same-origin reads of original audit images. Nothing is pre-fetched off-screen.
class ViewerResources {
 constructor(onStatus=()=>{}){
  this.onStatus=onStatus;this.pending=new Map();this.cache=new Map();this.bytes=0;
  this.busy=false;this.paused=false;this.retryAt=0;this.timer=null;this.lastStart=0;
  this.batchSupported=location.protocol!=='file:';this.idleWaiters=[];
  this.observer=new IntersectionObserver(()=>this.schedule(),{rootMargin:'160px'});
 }
 visible(im){
  if(!im.isConnected||im.closest('[hidden]')||!im.getClientRects().length)return false;
  const r=im.getBoundingClientRect();return r.bottom>=-160&&r.top<=innerHeight+160;
 }
 load(im,path){
  const url=new URL(path,location.href);
  if(url.origin!==location.origin){this.onStatus('图片地址必须与预览同源。');return}
  im.dataset.viewerSource=url.href;
  const cached=this.cache.get(url.href);
  if(cached){this.cache.delete(url.href);this.cache.set(url.href,cached);im.src=cached.url;this.pending.delete(im);this.observer.unobserve(im);return}
  // Remove an earlier frame immediately: old pixels must not appear under a new frame label.
  im.removeAttribute('src');im.width=im.width||256;im.height=im.height||256;
  this.pending.set(im,url.href);this.observer.observe(im);this.schedule();
 }
 schedule(delay=180){
  clearTimeout(this.timer);this.timer=setTimeout(()=>this.flush(),delay);
 }
 pause(response){
  const retry=response.headers.get('Retry-After');
  const seconds=retry!==null&&/^\d+$/.test(retry)?Number(retry):null;
  const until=seconds!==null?Date.now()+seconds*1000:Date.parse(retry||'');
  this.paused=true;this.retryAt=Number.isFinite(until)?Math.max(Date.now(),until):Date.now()+30000;
  this.onStatus(`转发服务限流（429）。已暂停资源请求，请稍后点击“重试加载”。`,this.retryAt);
 }
 retry(){
  if(Date.now()<this.retryAt){this.onStatus('仍在服务器要求的等待时间内，请稍后重试。',this.retryAt);return}
  this.paused=false;this.onStatus('');this.schedule(0);
 }
 whenIdle(){return !this.busy&&(this.paused||!this.hasVisiblePending())?Promise.resolve():new Promise(resolve=>this.idleWaiters.push(resolve))}
 hasVisiblePending(){return [...this.pending.keys()].some(im=>this.visible(im))}
 settle(){if(!this.busy&&(!this.hasVisiblePending()||this.paused))this.idleWaiters.splice(0).forEach(resolve=>resolve())}
 store(key,blob){
  const previous=this.cache.get(key);if(previous){URL.revokeObjectURL(previous.url);this.bytes-=previous.size;this.cache.delete(key)}
  this.cache.set(key,{url:URL.createObjectURL(blob),size:blob.size});this.bytes+=blob.size;
  while(this.bytes>64*1024*1024||this.cache.size>512){
   const [old,value]=this.cache.entries().next().value;this.cache.delete(old);this.bytes-=value.size;URL.revokeObjectURL(value.url);
  }
 }
 async flush(){
  if(this.busy||this.paused){this.settle();return}
  for(const im of this.pending.keys())if(!im.isConnected){this.pending.delete(im);this.observer.unobserve(im)}
  const keys=[...new Set([...this.pending].filter(([im])=>this.visible(im)).map(([,key])=>key))].slice(0,this.batchSupported?64:1);
  if(!keys.length){this.settle();return}
  const remaining=400-(Date.now()-this.lastStart);if(remaining>0){this.schedule(remaining);return}
  this.busy=true;this.lastStart=Date.now();
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),20000);
  try{
   const paths=keys.map(key=>{const u=new URL(key);return u.pathname+u.search});
   const response=await fetch(this.batchSupported?'/__harness_viewer__/images':keys[0],this.batchSupported?
    {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({paths}),signal:controller.signal}:
    {signal:controller.signal,cache:'force-cache'});
   if(response.status===429){this.pause(response);return}
   if(this.batchSupported&&[404,501].includes(response.status)){this.batchSupported=false;return}
   if(!response.ok)throw Error('HTTP '+response.status);
   if(this.batchSupported){
    if(!response.headers.get('Content-Type')?.startsWith('application/x-harness-images'))throw Error('批量资源响应格式不正确');
    const buffer=await response.arrayBuffer();if(buffer.byteLength<4)throw Error('图片数据不完整');
    const length=new DataView(buffer).getUint32(0),start=4+length;
    if(length>1024*1024||start>buffer.byteLength)throw Error('图片索引不完整');
    const manifest=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,length)));
    if(manifest.version!==1||manifest.images.length!==keys.length)throw Error('图片索引不匹配');
    for(let i=0;i<keys.length;i++){
     const entry=manifest.images[i];if(entry.path!==paths[i])throw Error('图片路径不匹配');
     if(entry.status!==200){
      for(const [im,key] of this.pending)if(key===keys[i]){im.alt+='（文件读取失败）';this.pending.delete(im);this.observer.unobserve(im)}
      this.onStatus(`图片读取失败（${entry.status}）：${paths[i]}`);continue;
     }
     if(!Number.isInteger(entry.offset)||!Number.isInteger(entry.size)||entry.offset<0||entry.size<0||start+entry.offset+entry.size>buffer.byteLength)throw Error('图片字节范围不正确');
     this.store(keys[i],new Blob([new Uint8Array(buffer,start+entry.offset,entry.size)],{type:entry.type}));
    }
   }else this.store(keys[0],await response.blob());
   for(const [im,key] of this.pending){const cached=this.cache.get(key);if(cached&&im.dataset.viewerSource===key){im.src=cached.url;this.pending.delete(im);this.observer.unobserve(im)}}
  }catch(error){this.paused=true;this.onStatus(`资源加载失败：${error.message}。可恢复预览连接后点击“重试加载”。`)}
  finally{clearTimeout(timeout);this.busy=false;if(!this.paused&&this.hasVisiblePending())this.schedule();this.settle()}
 }
}
window.ViewerResources=ViewerResources;
