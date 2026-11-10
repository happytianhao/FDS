'use strict';
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const directMemory=D.method==='sys2_memory'||D.summary?.method==='sys2_memory'||D.calls.some(c=>c.input.context.decision_protocol==='sys2_memory')||D.summary?.memory_enabled===true&&D.method==='sys2';
const isNew=D.method==='new'||D.summary?.method==='new';
const isV3=isNew||directMemory||D.calls.some(c=>c.input.context.decision_protocol==='tool_loop_v3');
const method=D.method||D.summary?.method||D.calls[0]?.input.context.decision_protocol;
const episodic=method==='sys1+2_memory';
const stateless=!directMemory&&['sys1+2','sys2'].includes(method);
const nativeControl=D.benchmark==='roboquest'||D.calls.some(c=>c.input.context.benchmark_control==='roboquest_native_v1');
const actionCount=key=>nativeControl?['arm','base','wait'].reduce((n,k)=>n+(D.diagnostics?.[key]?.[k]||0),0):(D.diagnostics?.[key]?.[actionTool]||0);
const actionTool=['sys2','sys2_memory','new'].includes(method)?'move_ee':'vla';
const hasEvidence=D.calls.some(c=>c.input.context.harness_revision==='evidence_v2');
const cats=isNew?{events:'事件进展与顺序',objects:'物体指代、位置与运动',experiences:'本回合经验'}:isV3?{events:'事件进展与顺序',objects:'物体指代、位置与运动',skills:'经验、技巧与偏好',...(hasEvidence?{evidence:'证据：关键帧与工具回执'}:{})}:{identity:'身份与指代',spatial:'空间关系',event:'事件与次数',procedure:'示范与顺序',perception:'短时感知',physical:'物理规律',experience:'失败与恢复经验',preference:'偏好与约定'};
const names={success:'官方成功',failure:'未成功',error:'运行错误',running:'执行中',not_started:'未启动',pending:'待运行',interrupted:'用户中断',not_run:'未运行（用户停止）',diagnostic_early_stop:'诊断性截断',post_terminal_review_interruption:'环境已结束，复核中断'};
const pretty=x=>JSON.stringify(x,null,2), val=x=>x===undefined||x===null?'未知':String(x);
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}
function badge(text,cls){return node('span',text,'badge '+(cls||''))}
function replace(id,...children){$(id).replaceChildren(...children)}
function links(id,items){replace(id,...Object.entries(items).map(([label,url])=>{const a=node('a',label);a.href=url;a.target='_blank';a.rel='noopener';return a}))}
function stat(label,value){const e=node('div',undefined,'stat');e.append(node('small',label),node('b',value));return e}
function decisionField(label,value){
 const e=node('div',undefined,'decision-field');e.append(node('h3',label));
 let structured=false;
 if(typeof value==='string'&&/^[\s]*[\[{]/.test(value)){try{JSON.parse(value);structured=true}catch{}}
 e.append(node(structured?'pre':'p',value||'—',structured?'decision-json':'call-detail'));return e;
}
function inputSkills(skills){
 const panel=$('skill-input');if(!panel)return;
 panel.hidden=!isNew;$('skill-title').textContent=`本轮输入的只读 skill（${skills.length} 条）`;
 replace('skill-list',...skills.map(skill=>{
  const item=node('details',undefined,'skill-item');item.append(node('summary',skill.id));
  item.append(node('p',skill.applicability,'muted'),node('p',skill.instruction,'call-detail'));
  if(skill.sources?.length)item.append(node('pre',pretty(skill.sources)));
  return item;
 }));
}
const s=D.summary||{}, last=D.calls.at(-1)||{}, lastAfter=last.after||{}, stateSteps=s.steps??lastAfter.step??last.before?.step??0;
replace('summary',stat(D.validation_only?'接口验证结果（非任务成绩）':'本条结果',D.validation_only?(s.error?'运行错误':s.stop_reason||'运行中'):names[D.status]),stat('环境步',stateSteps),stat('Sys2 调用',D.calls.length),stat(nativeControl?'原生动作块执行 / Sys2 选择':['sys2','sys2_memory','new'].includes(method)?'IK 动作块执行 / Sys2 选择':'VLA 执行块 / Sys2 选择',`${actionCount('executed_chunks_by_tool')} / ${actionCount('selected_tools')}`),stat(episodic?'记忆方式':'主动保存 / 回看 / 删除',episodic?'episode 内文字':stateless?'不启用':`${D.retention_counts.save||0} / ${D.retention_counts.read||0} / ${D.retention_counts.delete||0}`));
if(s.benchmark_metrics?.progress!==undefined)$('summary').append(stat('官方任务进度',`${(100*s.benchmark_metrics.progress).toFixed(1)}%`));

$('task-goal').textContent=D.calls[0]?.input.context.task_goal||s.task_goal||'尚未记录任务目标';
if(D.intervention){$('intervention-notice').hidden=false;$('intervention-notice').textContent=(D.status==='post_terminal_review_interruption'?`环境已在第 ${D.environment_result.step} 步结束，官方状态为 ${D.environment_result.official_status}；最终 Sys2 复核被操作员中断。整条运行仍保留原始错误，不能当作完整流水线成功。原因：`:'诊断性截断：不是完整 episode 成绩。原始收尾异常保留，不应直接视为自然发生的服务故障。原因：')+D.intervention.reason;}
$('protocol-note').textContent=isV3?`tool_loop_v3 / ${D.calls[0]?.input.context.harness_revision||'baseline'}：三类完整记忆；关键帧自动随每轮输入，直到删除。`:'旧版协议：历史媒体按显式读取工具请求。';
if(isNew){$('protocol-note').textContent='new：三类增量记忆（未修改条目保留）；关键帧显式增删；skill 在本回合只读；按相对执行步号选择下一轮观测并保留最终帧。';document.querySelector('main > .note').textContent='本页只读。记忆为模型估计，技能来自先前训练经历；环境官方评分独立记录。'}
if(hasEvidence)$('protocol-note').textContent='tool_loop_v3 / evidence_v2：四组完整记忆快照；证据含关键帧和工具回执，输出即保留，省略即移除。';
if(stateless){$('protocol-note').textContent=`${method}：无外部记忆、无历史会话。只输入新观测和上一段执行反馈；以下审计回放仅供人工分析。`;document.querySelector('main > .note').textContent='本页只读。新画面与动作只给 Sys2 一次；此方法不保存模型记忆。完整录像与审计快照仅供人类分析，模型无法从这里检索。官方评分与 Sys2 自我判断分别展示。'}
if(directMemory){$('protocol-note').textContent=(nativeControl?'sys2_memory：Sys2 使用 RoboQuest 官方 ARM/BASE/WAIT 控制器；':'sys2_memory：Sys2 直接输出 TCP 位姿并经 IK 执行；')+'三类模型记忆与主动选存关键帧。完整快照省略即删除，历史图像不代表当前状态。';document.querySelector('main > .note').textContent='本页只读。新画面与动作只输入一次；只有模型明确选存的关键帧可以回看。记忆是模型估计，官方评分独立展示。'}
if(episodic){for(const id of ['plan','memory-chart','retained'])$(id).closest('.panel').hidden=true;$('category').closest('.tabbar').hidden=true;$('memory-rows').closest('.scroll').hidden=true;$('protocol-note').textContent='Sys1+2 + episode 内文字记忆：模型主动维护完整快照；旧图片不可回看。记忆为模型估计。';document.querySelector('main > .note').textContent='每个 episode 清空记忆。新观测、上一段执行和主动保留的文字记忆送给模型；完整审计录像只供人工分析。'}
if(D.calls.some(c=>c.transport_retry?.attempts?.length>1))$('protocol-note').textContent+=' 本次含通信重试：同一逻辑轮次的输入可能重发，期间环境不动，成功决策只执行一次。请求尝试均单独记录。';
$('category').options[0].textContent=hasEvidence?'全部 4 类':isV3?'全部 3 类':'全部 8 类';
if(isV3)$('retained-title').textContent='主动保存的关键帧（自动随记忆输入）';
if(hasEvidence)$('retained-title').textContent='证据材料：关键帧与宿主工具回执';
for(const [k,v] of Object.entries(cats)){const o=node('option',v);o.value=k;$('category').append(o)}
links('episode-files',{...D.files,...(D.trace?{'trace.jsonl':D.trace}:{})});$('integrity').textContent=pretty(D.integrity);
let callIndex=0, frameIndex=0, timer=null;
let retryTimer=null;
const imageResources=new ViewerResources((message,retryAt=0)=>{
 $('resource-notice').hidden=!message;$('resource-message').textContent=message;
 clearTimeout(retryTimer);$('resource-retry').disabled=Date.now()<retryAt;
 if(retryAt>Date.now())retryTimer=setTimeout(()=>{$('resource-retry').disabled=false},Math.min(2147483647,retryAt-Date.now()));
});
$('resource-retry').onclick=()=>imageResources.retry();
function setImage(im,path){if(im.dataset.viewerSource===new URL(path,location.href).href)return;im.onerror=()=>{$('image-notice').hidden=false;$('image-notice').textContent='图片解码失败：'+im.dataset.viewerSource};imageResources.load(im,path)}
function photo(path,label,points=[]){const fig=node('figure'),a=node('a',undefined,'photo-link'),im=node('img');a.href=path;a.target='_blank';a.rel='noopener';im.alt=label;im.loading='lazy';setImage(im,path);a.append(im);for(const g of points){const dot=node('span',undefined,'point');dot.style.left=(g.point_yx[1]/10)+'%';dot.style.top=(g.point_yx[0]/10)+'%';dot.title=`${g.subject}: ${g.point_yx} (${g.confidence})`;dot.append(node('span',g.subject,'point-label'));a.append(dot)}fig.append(a,node('figcaption',label));return fig}
function inputPhoto(im,grounding){const strip=im.label.layout==='temporal_strip',frames=im.label.frames||[im.label.frame];const fig=photo(im.path,`${strip?'连续图，左→右':'帧'} ${frames.join(', ')} · ${im.label.camera} · ${im.label.source} · ${im.label.origin==='training_example'?'训练示例 · ':''}${im.label.cue?'演示':'执行'}`,strip?[]:grounding.filter(g=>im.label.camera==='front'&&g.frame===im.label.frame));if(strip)fig.classList.add('full');return fig}
function records(m){if(!m)return [];if(m.records)return m.records;return ['events','objects','experiences','skills','evidence'].flatMap(category=>(m[category]||[]).map(r=>({...r,key:category+':'+r.id,category,value:pretty(r),source:category==='evidence'?'host_record':'model_estimate',status:r.status||(category==='evidence'?'retained':undefined)})))}
function displayDecision(d){if(d?.tool_batch)d={...d,...d.tool_batch};return d?.tool_calls?{...d,tool:d.tool_calls.map(t=>t.name).join(' → ')}:d?.tool_call?{...d,...d.tool_call.arguments,tool:d.tool_call.name}:d||{}}
function memory(c){
 if(episodic){$('memory-before').textContent=pretty(c.before.episodic_memory);$('memory-after').textContent=pretty(c.after.episodic_memory);replace('memory-stats',node('p','模型记忆完整快照；宿主未验证语义。'),badge(`事件记录条目 ${c.before.episodic_memory?.events?.length||0} → ${c.after.episodic_memory?.events?.length||0}`));$('memory-ops').textContent=pretty(c.events.filter(e=>e.event==='episodic_memory_update'));return}
 if(stateless){$('memory-before').textContent=$('memory-after').textContent='此方法不维护外部记忆。';replace('memory-stats',node('p','当前可见记录只用于人工审计，不会回送给 Sys2。'));return}
 const before=records(c.before?.memory),after=records(c.after?.memory);
 const b=new Map(before.map(r=>[r.key,r])),a=new Map(after.map(r=>[r.key,r]));
 const seen=new Set([...records(c.input.context.memory),...(c.input.context.memory_read_result?.records||[])].map(r=>r.key));
 const rows=[],counts={add:0,delete:0,update:0,same:0};
 for(const key of new Set([...b.keys(),...a.keys()])){
  const old=b.get(key),r=a.get(key)||old,change=!old?'add':!a.has(key)?'delete':pretty(old)!==pretty(r)?'update':'same';counts[change]++;
  if($('category').value&&r.category!==$('category').value||$('changes-only').checked&&change==='same')continue;
  const tr=node('tr'),id=node('td'),status=node('td'),content=node('td'),evidence=node('td');
  id.append(node('strong',key),node('div',cats[r.category]||r.category,'muted'));
  status.append(badge({add:'新增',delete:'删除',update:'更新',same:'未变'}[change],change));status.append(node('div',seen.has(key)?'本轮已输入':'本轮未输入','muted'));
  content.append(node('div',r.value,'value'));if(change==='update'){const d=node('details'),p=node('pre',pretty(old));d.append(node('summary','查看调用前内容'),p);content.append(d)}
  const full=node('details');full.append(node('summary','完整记录字段'),node('pre',pretty(r)));content.append(full);
  evidence.append(node('div',`${r.status} · ${r.category==='evidence'?'宿主原始材料（描述由模型提供）':isV3?'模型记忆（非当前观测）':r.stale?'过时':'有效'} · ${r.source||'model_estimate'}`),node('div',r.category==='evidence'?(r.kind==='frame'?`原始帧 ${r.frame_id}`:`工具回执 ${r.call_id}；不代表物理成功`):`帧 ${(r.provenance_frames||r.evidence_frames||[]).join(', ')} | ${isV3?'最后观察帧 '+(r.last_seen_frame??'—'):'有效至 step '+(r.expires_step??'∞')}`,'muted'));
  if(r.position?.space&&r.position.space!=='none')evidence.append(node('div',`${r.position.space}: ${r.position.coordinates?.join(', ')||r.position.relation||'未知'} · 帧 ${r.position.frame}`));
  if(r.condition)evidence.append(node('div','条件：'+r.condition));tr.append(id,status,content,evidence);rows.push(tr);
 }
 replace('memory-rows',...rows);replace('memory-stats',badge(`之前 ${before.length} → 之后 ${after.length}`),badge(`新增 ${counts.add}`,'add'),badge(`更新 ${counts.update}`,'update'),badge(`删除 ${counts.delete}`,'delete'));
 $('memory-before').textContent=pretty({memory:c.before.memory,retained:c.before.retained,evidence_materials:c.before.evidence_materials,scene:c.before.scene});$('memory-after').textContent=pretty({memory:c.after.memory,retained:c.after.retained,evidence_materials:c.after.evidence_materials,scene:c.after.scene});
 const memoryEvents=c.events.filter(e=>['memory_operation','automatic_demo_memory'].includes(e.event));
 const rejected=memoryEvents.flatMap(e=>e.record_results||[]).filter(r=>r.status==='rejected');
 for(const r of rejected)$('memory-stats').append(node('p',`${r.group}/${r.id}：本轮更新被拒，${r.resolution==='kept_previous'?'保留旧记录':'未新增记录'}。${r.error}`,'muted'));
 $('memory-ops').textContent=pretty(memoryEvents);
 const rb=new Map((c.before.retained?.catalog||[]).map(r=>[r.key,r])),ra=new Map((c.after.retained?.catalog||[]).map(r=>[r.key,r]));
 const cards=[];
 for(const key of new Set([...rb.keys(),...ra.keys()])){
  const r=ra.get(key)||rb.get(key),change=!rb.has(key)?'add':!ra.has(key)?'delete':pretty(rb.get(key))!==pretty(r)?'update':'same',card=node('div',undefined,'card');
  card.append(node('h3',key),badge({add:'本轮保存',delete:'本轮删除',update:'本轮替换',same:'已保存'}[change],change),node('p',r.description),node('small','理由：'+r.reason));
  const storage=(ra.has(key)?c.after.retained?.storage:c.before.retained?.storage)?.[key],file=D.retained_files?.[storage];if(storage){const location=node(file?'a':'p',`retained/${storage}/content.json${file?'':'（已删除或替换，以下为历史审计）'}`,'files');if(file){location.href=file;location.target='_blank';location.rel='noopener'}card.append(location)}
  const gallery=node('div',undefined,'gallery');for(const fid of r.frame_ids){const f=D.frames.find(f=>f.frame===fid);if(f)for(const [camera,path] of Object.entries(f.images))gallery.append(photo(path,`保存的帧 ${fid} · ${camera} · ${f.cue?'演示':'执行'}`))}card.append(gallery,node('pre',pretty({frame_ids:r.frame_ids,segment_ids:r.segment_ids,size:r.size,saved_call:r.saved_call})));
  const evidence=node('details'),segments=D.calls.map(c=>c.input.context.latest_execution).filter(s=>s&&r.segment_ids.includes(s.segment_id));evidence.append(node('summary','选存的状态与实际动作（审计副本）'),node('pre',pretty({frames:D.frames.filter(f=>r.frame_ids.includes(f.frame)),segments})));card.append(evidence);cards.push(card);
 }
 const tb=c.before.evidence_materials?.tool_results||{},ta=c.after.evidence_materials?.tool_results||{};
 for(const key of new Set([...Object.keys(tb),...Object.keys(ta)])){const receipt=ta[key]||tb[key],change=!tb[key]?'add':!ta[key]?'delete':pretty(tb[key])!==pretty(ta[key])?'update':'same',card=node('div',undefined,'card');card.append(node('h3',key),badge({add:'本轮保留',delete:'本轮省略',update:'本轮替换',same:'继续保留'}[change],change),node('p','宿主工具回执：证明调用和执行情况，物理成功仍未验证。'),node('pre',pretty(receipt)));cards.push(card)}
 replace('retained',...(cards.length?cards:[node('p','这一轮前后都没有主动保留的原始记忆。','empty')]));
 $('retained-ops').textContent=pretty({operations:c.events.filter(e=>['retention_operation','retained_read','evidence_snapshot'].includes(e.event)),read_result:c.input.context.evidence_materials||c.input.context.retained_read_result||c.input.context.keyframe_memory});
 memoryChart();
}
function memoryChart(){
 const svg=$('memory-chart');svg.replaceChildren();const counts=D.calls.map(c=>[records(c.after.memory||c.before.memory).length,c.after.retained?.catalog?.length??c.before.retained?.catalog?.length??0]);
 if(!counts.length)return;const max=Math.max(1,...counts.flat()),x=i=>45+i/Math.max(1,counts.length-1)*830,y=n=>172-n/max*140;
 const make=(tag,attrs)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);svg.append(e);return e};
 for(const [i,color]of ['#087f8c','#ad597e'].entries()){make('polyline',{points:counts.map((v,k)=>`${x(k)},${y(v[i])}`).join(' '),fill:'none',stroke:color,'stroke-width':2});const t=make('text',{x:50+i*200,y:17,fill:color,'font-size':12});t.textContent=i?'主动选存条目':'语义记录条数'}
 make('line',{x1:x(callIndex),x2:x(callIndex),y1:25,y2:174,stroke:'#63758a','stroke-dasharray':'4 4'});
 const t=make('text',{x:45,y:196,fill:'#62768a','font-size':12});t.textContent=`调用 1 → ${counts.length} · 纵轴 0 → ${max} 条`;
}
function chart(id,dim){
 const svg=$(id);svg.replaceChildren();const actions=D.actions;
 const make=(tag,attrs)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);svg.append(e);return e};
 if(!actions.length||dim===7&&actions.some(a=>a.action.length<8)){const t=make('text',{x:30,y:100,'font-size':15,fill:'#62768a'});t.textContent=!actions.length?'尚无实际执行动作':'当前动作空间没有夹爪维度';return}
 const dims=dim===7?[7]:[0,1,2,3,4,5,6],values=actions.flatMap(a=>dims.map(d=>a.action[d]));let lo=Math.min(...values),hi=Math.max(...values);if(hi-lo<.01){lo-=.5;hi+=.5}
 const xmin=actions[0].step,xmax=actions.at(-1).step,x=n=>46+(n-xmin)/Math.max(1,xmax-xmin)*835,y=n=>173-(n-lo)/(hi-lo)*148;
 const c=D.calls[callIndex];if(c){const from=c.before.step??0,to=c.after.step??from;make('rect',{x:x(Math.max(xmin,from)),y:16,width:Math.max(0,x(to)-x(Math.max(xmin,from))),height:160,fill:'#d8ece9'})}
 const colors=['#087f8c','#ef8354','#6659ae','#d34c76','#58955c','#b68e16','#3975ba','#213d62'];
 for(const d of dims){make('polyline',{points:actions.map(a=>`${x(a.step)},${y(a.action[d])}`).join(' '),fill:'none',stroke:colors[d],'stroke-width':1.5});const t=make('text',{x:56+d%7*92,y:13,fill:colors[d],'font-size':11});t.textContent=d===7?'gripper':`joint ${d+1}`}
 for(const [value,py]of[[hi,26],[lo,176]]){const t=make('text',{x:2,y:py,'font-size':11,fill:'#62768a'});t.textContent=value.toFixed(2)}
 const t=make('text',{x:46,y:195,'font-size':11,fill:'#62768a'});t.textContent=`step ${xmin} → ${xmax}`;
}
function frame(index,seek=true){
 if(!D.frames.length){$('frame-label').textContent='没有已记录画面';return}frameIndex=Math.max(0,Math.min(index,D.frames.length-1));$('frame-slider').value=frameIndex;
 const f=D.frames[frameIndex];$('frame-label').textContent=`帧 ${f.frame} · step ${f.step} · ${f.cue?'演示':'执行'}`;
 if(!$('execution').hidden){if(videoMode()){if(seek){stopPlayback();seekVideo(frameIndex)}}else{
  for(const camera of ['front','right','wrist']){if(f.images[camera]){setImage($(camera),f.images[camera]);$(camera).hidden=false}else $(camera).hidden=true}
  if($('right-camera'))$('right-camera').hidden=!f.images.right;
 }}
 const shown=[];for(const c of D.calls){const sources=new Set(c.input.images.filter(i=>(i.label.frames||[i.label.frame]).includes(f.frame)).map(i=>i.label.source));if(sources.size)shown.push(`#${c.number} ${[...sources].map(x=>x==='fresh'?'首次输入':x==='current_repeat'?'当前状态重发':x==='window_review'?'当前窗口补看':x==='training_example'?'额外训练示例':'已保存回看').join('/')}`)}
 $('frame-visibility').textContent=shown.length?'Sys2 可见记录：'+shown.join('，'):'此帧只在完整审计中，未输入 Sys2。';$('frame-state').textContent=pretty(f.robot_state);
}
const rawTextCache=new Map();
function rawText(url,fallback){
 if(!url)return Promise.resolve(fallback);
 if(imageResources.paused)return Promise.reject(Error('资源请求已暂停，请先重试加载'));
 if(!rawTextCache.has(url))rawTextCache.set(url,fetch(url).then(response=>{if(response.status===429)imageResources.pause(response);if(!response.ok)throw Error('HTTP '+response.status);return response.text()}).catch(error=>{rawTextCache.delete(url);throw error}));
 return rawTextCache.get(url);
}
async function loadRaw(c){
 const targets=[['prompt','prompt.txt',''],['response','response.json','尚无原始响应']];
 await Promise.all(targets.map(async([id,file,fallback])=>{
  let text;try{text=await rawText(c.files[file],fallback)}catch(error){text='无法读取原始文件：'+error.message+' · '+c.files[file]}
  if(D.calls[callIndex]===c)$(id).textContent=text||fallback;
 }));
 if(D.calls[callIndex]===c)$('raw').dataset.loadedCall=String(c.number);
}
function select(index){
 stopPlayback();
 if(!D.calls.length){$('call-title').textContent='尚无 Sys2 请求';$('prev').disabled=$('next').disabled=true;return}
 callIndex=Math.max(0,Math.min(index,D.calls.length-1));const c=D.calls[callIndex],ctx=c.input.context,d=displayDecision(c.decision);
 $('call-slider').value=callIndex;$('call-title').textContent=`第 ${c.number} / ${D.calls.length} 轮`;$('prev').disabled=callIndex===0;$('next').disabled=callIndex===D.calls.length-1;
 document.querySelectorAll('#call-list button').forEach((b,i)=>b.classList.toggle('active',i===callIndex));
 replace('call-meta',badge(`step ${c.before.step??'?'} → ${c.after.step??'执行中'}`),badge(`工具 ${d.tool||'等待响应'}`),badge(`输入图 ${c.input.images.length}`),badge(`模型耗时 ${c.stats.seconds?.toFixed(1)??'未知'} s`),badge(`tokens ${val(c.stats.usage?.input_tokens??c.stats.usage?.prompt_tokens)} / ${val(c.stats.usage?.output_tokens??c.stats.usage?.completion_tokens)}`));links('files',c.files);
 if(c.transport_retry?.attempts?.length)$('call-meta').append(badge(`请求尝试 ${c.transport_retry.attempts.length} 次 · 接受 ${c.transport_retry.accepted_attempt??'无'}`));
 $('delivery').textContent=`阶段：${ctx.observation?.phase}；当前帧 ${ctx.current_frame}。fresh 为新观测，history_review 为主动请求的历史回看，retained_memory/retained_history 为主动保留的关键帧。旧记录中的 window_review/current_repeat 分别表示旧协议的窗口补看/当前帧重发。training: 帧始终来自额外训练示例，不能当作当前任务进展。红点是本轮 Sys2 输出的定位叠加，原始输入没有红点；点击图片查看原图。`;
 replace('gallery',...(c.input.images.length?c.input.images.map(im=>inputPhoto(im,d.grounding||[])):[node('p','本轮没有图像输入；辅助调用、零步拒绝或决策纠正不会自动重放旧观测。','empty')]));
 inputSkills(ctx.skills||[]);
 $('latest').textContent=pretty({previous_subgoal:ctx.previous_subgoal,memory_validation:ctx.memory_validation,training_example:ctx.training_example,last_tool_result:ctx.last_tool_result,benchmark_control:ctx.benchmark_control,robot_state_layout:ctx.robot_state_layout,current_robot:ctx.current_robot,robot_geometry:ctx.robot_geometry,kinematics:ctx.kinematics,pose_limits:ctx.pose_limits,coordinates:ctx.coordinates,decision_support:ctx.decision_support,latest_execution:ctx.latest_execution,last_tool_receipt:ctx.last_tool_receipt,robot_state_frames:ctx.observation?.frames,observation_sampling:ctx.observation?.sampling,observation_review_window:ctx.observation?.review_window,observation_review_windows:ctx.observation?.review_windows,observation_review_result:ctx.observation?.review_result,demonstration_sampling:ctx.observation?.demo_sampling,current_rgb_regions:ctx.observation?.perception,fresh_rgb_region_sequence:ctx.observation?.color_region_sequence,retained_segments:ctx.retained_read_result?.segments});
 const fixedCriterion=(c.after.plan?.nodes||c.before.plan?.nodes||[]).find(n=>n.id===d.subtask_id)?.success_criteria;
 const fields=isNew?[...(d.kind?[['工具批次类别',d.kind==='environment'?'环境工具':'辅助工具']]:[]),...(d.kind==='auxiliary'?[]:[['环境批次子目标',d.subgoal],...(d.observation_steps!==undefined?[['执行后观测相对步号（旧协议）',pretty(d.observation_steps)]]:[])]),['顺序工具调用',pretty(d.tool_calls)],['记忆增量',pretty(d.memory_changes)],['关键帧改动',pretty(d.keyframe_changes)]]:episodic?[['工具调用',pretty(d.tool_call)],['模型文字记忆',pretty(d.episodic_memory)]]:stateless?[['工具调用',pretty(d.tool_call)],['输入与执行约定','无外部记忆；位姿/关节单位、坐标系及动作范围见原始输入。']]:isV3?[['工具调用',pretty(d.tool_call)],['事件验证与进展',pretty((d.working_memory||d.memory)?.events)],[hasEvidence?'证据完整快照':'关键帧操作',pretty(directMemory?d.retained_frames:hasEvidence?d.memory?.evidence:d.keyframe_ops)]]:[['场景理解',d.scene_summary],['工具 / 子目标',`${d.tool||'等待响应'} / ${d.subtask_id||''}\n${d.subgoal||''}`],['本轮声明的成功标准',d.success_criteria],['计划中固定的成功标准',fixedCriterion],['验证上一动作',pretty(d.verification)],['定位 GroundSG',pretty(d.grounding)],['失败恢复',pretty(d.recovery)]];
 replace('decision',...fields.map(([label,value])=>decisionField(label,value)));
 $('feedback').textContent=pretty(c.outcome);$('decision-json').textContent=pretty(c.decision);memory(c);
 const p=isV3?{nodes:c.after.memory?.events||c.before.memory?.events||[]}:c.after.plan||c.before.plan||{},pn=Array.isArray(p.nodes)?p.nodes:Object.values(p.nodes||{});
 const groups=new Map();if(hasEvidence)for(const n of pn){if(n.count_group&&n.occurrence>0){if(!groups.has(n.count_group))groups.set(n.count_group,new Set());if(n.origin==='execution'&&n.status==='succeeded')groups.get(n.count_group).add(n.occurrence)}}
 replace('plan-counts',...(hasEvidence?[node('p','以下来自本轮已接受的记忆；完成次数是模型估计，不是官方评分。未接受的提案和备注中的数字不计入。','muted'),...Array.from(groups,([group,done])=>badge(`${group}：模型标为完成 ${done.size} 次 · 序号 ${[...done].sort((a,b)=>a-b).join(', ')||'无'}`))]:[]));
 replace('plan',...pn.map(n=>{const e=node('div',undefined,'card');e.append(badge(n.status,n.status),node('strong',`${n.id}: ${n.objective}`),node('p',`验证标准：${n.success_criteria}`));if(isV3){e.append(node('small',`来源 ${n.origin} · 执行编号 ${n.attempt_id||'尚未执行'}${hasEvidence?' · '+(n.count_group?`计数组 ${n.count_group} / 第 ${n.occurrence} 次`:'独立事件，不计重复次数'):''}`),node('p',n.note||'无备注'));if(Array.isArray(n.depends_on))e.append(node('small',`前置事件：${n.depends_on.join(', ')||'无（独立事件）'}`))}else e.append(node('small',`依赖 ${(n.depends_on||[]).join(', ')||'无'} · 尝试 ${n.attempts||0}`));return e}));$('plan-json').textContent=pretty(p);
 $('prompt').textContent='打开本页签后加载原始 Prompt';$('response').textContent='打开本页签后加载原始响应';delete $('raw').dataset.loadedCall;if(!$('raw').hidden)loadRaw(c);$('events').textContent=pretty({stats:c.stats,transport_retry:c.transport_retry,events:c.events,provider_events:c.provider_events});
 $('actions').textContent=pretty(c.events.filter(e=>e.event==='execution_outcome'));const native=c.input.context.benchmark_control==='roboquest_native_v1';$('joint-chart').hidden=native;$('gripper-chart').hidden=native;if(native){$('action-description').textContent='RoboQuest 原生 ARM/BASE/WAIT 工具；下方显示执行反馈。逐步记录为末端/底盘控制目标，底层动作另存于 official/physical-trace-private.jsonl。'}else{chart('joint-chart',0);chart('gripper-chart',7);}
 const fid=c.after.current_frame??c.before.current_frame,fi=D.frames.findIndex(f=>f.frame===fid);if(fi>=0)frame(fi);
 location.hash=`call-${c.number}`;
}
for(const c of D.calls){const b=node('button',c.number);b.title=`${displayDecision(c.decision).tool||'等待响应'} · step ${c.before.step}`;b.onclick=()=>select(c.number-1);$('call-list').append(b)}
let callDrag=null,frameDrag=null;
$('call-slider').max=Math.max(0,D.calls.length-1);$('call-slider').oninput=e=>{clearTimeout(callDrag);const index=+e.target.value;callDrag=setTimeout(()=>select(index),220)};$('prev').onclick=()=>select(callIndex-1);$('next').onclick=()=>select(callIndex+1);
$('category').onchange=$('changes-only').onchange=()=>{if(D.calls.length)memory(D.calls[callIndex])};
$('frame-slider').max=Math.max(0,D.frames.length-1);$('frame-slider').oninput=e=>{clearTimeout(frameDrag);const index=+e.target.value;frameDrag=setTimeout(()=>frame(index),220)};
const replay=$('replay-video'), replayInfo=D.replay_video||{};
let videoAvailable=replayInfo.status==='ready', pendingVideoTime=null;
function videoMode(){return videoAvailable&&$('replay-mode').value==='video'}
function stopPlayback(){if(timer){clearInterval(timer);timer=null}replay.pause();$('play').textContent='播放'}
function seekVideo(index){
 pendingVideoTime=(index+.01)/replayInfo.fps;
 if(replay.readyState>=1){replay.currentTime=Math.min(pendingVideoTime,replay.duration);pendingVideoTime=null}
}
function showReplayMode(){
 replay.hidden=!videoMode();$('replay-images').hidden=videoMode();$('replay-speed').disabled=!videoMode();
}
function loadVideo(){if(videoMode()&&!replay.getAttribute('src')){replay.preload='metadata';replay.src=replayInfo.url}}
function videoPosition(seconds){
 if(!videoMode()||replay.seeking)return;
 const index=Math.min(D.frames.length-1,Math.max(0,Math.floor(seconds*replayInfo.fps+.0001)));
 if(index!==frameIndex)frame(index,false);
}
if(videoAvailable){
 replay.preload='none';replay.muted=true;
 $('replay-download').href=replayInfo.url;
 $('replay-notice').textContent=`按 ${replayInfo.fps} fps 连续播放全部 ${replayInfo.frames} 帧，多个相机同步显示；包含演示，不包含模型等待时间。滑块与 Sys2 轮次可定位视频，逐帧原图可查看无损图片。`;
 replay.addEventListener('loadedmetadata',()=>{if(pendingVideoTime!==null){replay.currentTime=Math.min(pendingVideoTime,replay.duration);pendingVideoTime=null}});
 replay.addEventListener('play',()=>{$('play').textContent='暂停'});
 replay.addEventListener('pause',()=>{$('play').textContent='播放'});
 replay.addEventListener('ended',()=>{frame(D.frames.length-1,false);$('play').textContent='播放'});
 replay.addEventListener('seeked',()=>videoPosition(replay.currentTime));
 if(replay.requestVideoFrameCallback){
  const tick=(now,meta)=>{videoPosition(meta.mediaTime);replay.requestVideoFrameCallback(tick)};
  replay.requestVideoFrameCallback(tick);
 }else replay.addEventListener('timeupdate',()=>videoPosition(replay.currentTime));
 replay.addEventListener('error',()=>{stopPlayback();videoAvailable=false;$('replay-mode').value='images';$('replay-mode').disabled=true;showReplayMode();frame(frameIndex);$('replay-notice').textContent='视频加载失败，已切回逐帧原图。可尝试下载 MP4 或检查视频地址。'});
}else{
 $('replay-mode').value='images';$('replay-mode').disabled=true;$('replay-download').hidden=true;
 $('replay-notice').textContent=(replayInfo.reason||'尚未合成视频，请重新生成可视化。')+' 仍可逐帧查看原图。';
}
if(D.frame_scope==='model_observation_boundaries_only')$('replay-notice').textContent+=' 仅含模型观测时间点；没有补造动作之间的画面。';
$('replay-mode').onchange=()=>{stopPlayback();showReplayMode();loadVideo();frame(frameIndex)};
$('replay-speed').onchange=()=>{replay.playbackRate=Number($('replay-speed').value)};
$('play').onclick=()=>{
 if(!D.frames.length)return;
 if(videoMode()){
  loadVideo();
  if(!replay.paused){stopPlayback();return}
  if(frameIndex>=D.frames.length-1)frame(0);
  replay.play().catch(()=>{$('play').textContent='播放';$('replay-notice').textContent='视频暂时无法播放，请使用视频自带播放按钮或下载 MP4。'});
 }else if(timer)stopPlayback();else{
  if(frameIndex>=D.frames.length-1)frame(0);
  let advancing=false;
  timer=setInterval(async()=>{if(advancing||imageResources.paused)return;if(frameIndex>=D.frames.length-1){stopPlayback();return}advancing=true;frame(frameIndex+1,false);await imageResources.whenIdle();advancing=false},1000);
  $('play').textContent='暂停';
 }
};
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{if(b.dataset.tab!=='execution')stopPlayback();document.querySelectorAll('[data-tab]').forEach(n=>n.classList.toggle('active',n===b));document.querySelectorAll('.tabpage').forEach(n=>n.hidden=n.id!==b.dataset.tab);if(b.dataset.tab==='raw'&&D.calls.length)loadRaw(D.calls[callIndex]);if(b.dataset.tab==='execution'){loadVideo();frame(frameIndex)}imageResources.schedule()});
showReplayMode();
const initial=Number(location.hash.replace('#call-',''))-1;frame(0);select(Number.isFinite(initial)&&initial>=0?initial:0);document.body.dataset.ready='true';
