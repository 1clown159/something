/**
 * SEG-Y Viewer v9
 */
var $=function(id){return document.getElementById(id);};
var sgyMeta=null,svNav,heatmapData=null,heatmapVmin=0,heatmapVmax=1;

document.addEventListener('DOMContentLoaded',function(){
    svNav=new StepNavigation({steps:['upload-meta','heatmap','waveform','bitstats'],startStep:0,onStepChange:function(){}});
    var b=$('sgyUploadBtn');if(b)b.addEventListener('click',function(){$('sgyFileInput').click();});
    var f=$('sgyFileInput');if(f)f.addEventListener('change',handleUpload);
    var h=$('loadHeatmapBtn');if(h)h.addEventListener('click',loadHeatmap);
    var w=$('drawWaveBtn');if(w)w.addEventListener('click',drawWaveform);
    var bs=$('runBitStatsBtn');if(bs)bs.addEventListener('click',runBitStats);
});

function switchSgyView(view){
    $('viewHeatmap').style.display = view==='heatmap'?'block':'none';
    $('viewGray').style.display = view==='gray'?'block':'none';
    $('tabHeatmap').classList.toggle('active', view==='heatmap');
    $('tabGray').classList.toggle('active', view==='gray');
}

function handleUpload(e){
    var file=e.target.files[0];if(!file)return;
    $('sgyStatus').textContent='解析中...';$('sgyMetaResult').innerHTML='<div class="loading-spinner" style="margin:auto;"></div>';
    var fd=new FormData();fd.append('file',file);
    axios.post(API_BASE_URL+'/api/demo/parse-sgy',fd,{headers:{'Content-Type':'multipart/form-data'}}).then(function(resp){
        sgyMeta=resp.data;$('sgyStatus').textContent='已解析: '+file.name+' ('+sgyMeta.file_size_mb+' MB)';
        renderMeta(sgyMeta);$('sgyFileName').textContent=file.name;$('sgyFileSize').textContent=sgyMeta.file_size_mb+' MB';
        $('taskInfo').style.display='block';$('profileRange').textContent='范围: 0 – '+((sgyMeta.profile_count||1)-1);
        $('profileIdx').max=(sgyMeta.profile_count||1)-1;svNav.setStepEnabled(1,true);svNav.markStepDone(0);svNav.goToStep(1);
    }).catch(function(err){$('sgyStatus').textContent='解析失败';$('sgyMetaResult').innerHTML='<div class="chart-placeholder">'+getErr(err)+'</div>';});
}
function renderMeta(d){
    var rows=[["文件大小",d.file_size_mb+' MB'],["数据格式",d.format_name],["道数",d.trace_count.toLocaleString()],
        ["每道采样点",d.sample_count],["采样间隔",d.sample_interval_us+' μs'],["每道字节",d.trace_total_bytes+' B'],
        ["剖面数",d.profile_count||'?'],["每剖面道数",d.traces_per_profile||'?'],["推断维度",d.inferred_dimensions],["总样本数",d.total_samples.toLocaleString()]];
    if(d.inline_min!==null)rows.push(["Inline 范围",d.inline_min+' – '+d.inline_max]);
    if(d.crossline_min!==null)rows.push(["Crossline 范围",d.crossline_min+' – '+d.crossline_max]);
    var html='<table class="verify-table" style="margin:0;">';
    rows.forEach(function(r){html+='<tr><td style="color:var(--text-muted);width:180px;">'+r[0]+'</td><td style="font-weight:600;">'+r[1]+'</td></tr>';});
    html+='</table>';$('sgyMetaResult').innerHTML=html;
}

function loadHeatmap(){
    if(!sgyMeta||!sgyMeta.stored_path){alert('请先上传并解析 SEG-Y 文件');return;}
    var idx=parseInt($('profileIdx').value)||0;showLoading(true);
    axios.get(API_BASE_URL+'/api/sgy/heatmap',{params:{file_path:sgyMeta.stored_path,inline_idx:idx, max_cols:500, max_rows:600}}).then(function(resp){
        var d=resp.data;heatmapData=d.data;heatmapVmin=d.vmin;heatmapVmax=d.vmax;renderCanvas(d,idx);
    }).catch(function(err){$('heatmapContainer').innerHTML='<div class="chart-placeholder">加载失败: '+getErr(err)+'</div>';}).finally(function(){showLoading(false);});
}

function renderCanvas(result,idx){
    var data=result.data,cc=data.length,rc=data[0]?data[0].length:0;if(cc===0||rc===0)return;
    var cellH=1,cellW=2,aw=cc*cellW,ah=rc*cellH;var c=$('heatmapContainer');c.innerHTML='';
    var canvas=document.createElement('canvas');canvas.width=aw;canvas.height=ah;
    canvas.style.cssText='display:block;max-width:100%;height:auto;margin:0 auto;cursor:crosshair;image-rendering:pixelated;';c.appendChild(canvas);
    var ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;
    var vMin=result.vmin!==undefined?result.vmin:heatmapVmin,vMax=result.vmax!==undefined?result.vmax:heatmapVmax,vR=vMax-vMin||1;
    function hc(t){t=Math.max(0,Math.min(1,t));var r,g,b;if(t<0.5){var s=t*2;r=Math.round(33+s*222);g=Math.round(62+s*193);b=Math.round(181+s*74);}else{var s=(t-0.5)*2;r=255;g=Math.round(255-s*220);b=Math.round(255-s*220);}return[r,g,b];}
    var imgData=ctx.createImageData(aw,ah),buf=imgData.data;
    for(var tr=0;tr<cc;tr++){var row=data[tr];if(!row)continue;for(var sm=0;sm<row.length;sm++){var cl=hc((row[sm]-vMin)/vR),py=sm*cellH,px=tr*cellW;for(var dx=0;dx<cellW;dx++){var bi=(py*aw+(px+dx))*4;buf[bi]=cl[0];buf[bi+1]=cl[1];buf[bi+2]=cl[2];buf[bi+3]=255;}}}
    ctx.putImageData(imgData,0,0);
    var barW=14,barH=Math.min(ah,400),lC=document.createElement('canvas');lC.width=barW+30;lC.height=barH;lC.style.cssText='display:block;';var lx=lC.getContext('2d');
    for(var s=0;s<barH;s++){var cl2=hc(1-s/barH);lx.fillStyle='rgb('+cl2[0]+','+cl2[1]+','+cl2[2]+')';lx.fillRect(0,s,barW,1);}
    lx.fillStyle='#666';lx.font='9px Inter';lx.fillText(vMax.toFixed(1),barW+3,10);lx.fillText(vMin.toFixed(1),barW+3,barH-2);
    var wrap=document.createElement('div');wrap.style.cssText='display:flex;align-items:flex-start;gap:8px;';wrap.appendChild(canvas);wrap.appendChild(lC);c.appendChild(wrap);
    var tip=null;
    canvas.addEventListener('mousemove',function(e){var rect=canvas.getBoundingClientRect(),mx=Math.floor((e.clientX-rect.left)/cellW),my=Math.floor((e.clientY-rect.top)/cellH);if(mx>=0&&mx<cc&&data[mx]&&my>=0&&my<data[mx].length){var val=data[mx][my];if(!tip){tip=document.createElement('div');tip.className='sgy-tip';tip.style.cssText='position:fixed;background:#fff;color:#1a1a2e;padding:4px 8px;border-radius:4px;font-size:11px;font-family:JetBrains Mono,monospace;pointer-events:none;z-index:999;border:1px solid rgba(0,0,0,0.15);box-shadow:0 2px 8px rgba(0,0,0,0.1);';document.body.appendChild(tip);}tip.style.display='block';tip.textContent='道#'+mx+' 样点#'+my+' = '+val.toFixed(4);tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-24)+'px';}else if(tip){tip.style.display='none';}});
    canvas.addEventListener('mouseleave',function(){if(tip)tip.style.display='none';});
    svNav.setStepEnabled(2,true);svNav.setStepEnabled(3,true);
    if($('traceIdx')){$('traceIdx').max=cc-1;$('traceRange').textContent='范围: 0 – '+(cc-1);}
    $('heatmapInfo').textContent='剖面 #'+idx+' | '+cc+' 道 × '+rc+' 采样 | 值域 ['+vMin.toFixed(2)+', '+vMax.toFixed(2)+'] | W:H='+aw+':'+ah;

    // ===== Grayscale image =====
    renderGrayscale(data, cc, rc, vMin, vMax, idx);
}
function renderGrayscale(data, cc, rc, vMin, vMax, idx){
    var gCellW=2, gCellH=1, gw=cc*gCellW, gh=rc*gCellH;
    var gc=$('grayContainer'); gc.innerHTML='';
    var gCanvas=document.createElement('canvas'); gCanvas.width=gw; gCanvas.height=gh;
    gCanvas.style.cssText='display:block;max-width:100%;height:auto;margin:0 auto;cursor:crosshair;image-rendering:pixelated;';
    gc.appendChild(gCanvas);
    var gCtx=gCanvas.getContext('2d'); gCtx.imageSmoothingEnabled=false;
    var vR=vMax-vMin||1;
    var gImg=gCtx.createImageData(gw,gh), gBuf=gImg.data;
    for(var t=0;t<cc;t++){var row=data[t];if(!row)continue;for(var s=0;s<row.length;s++){
        var val=(row[s]-vMin)/vR, gs=Math.round(Math.max(0,Math.min(1,val))*255);
        var py=s*gCellH, px=t*gCellW;
        for(var dx=0;dx<gCellW;dx++){var bi=(py*gw+(px+dx))*4;gBuf[bi]=gs;gBuf[bi+1]=gs;gBuf[bi+2]=gs;gBuf[bi+3]=255;}
    }}
    gCtx.putImageData(gImg,0,0);

    // ===== Observation box on grayscale =====
    var obsBoxW=Math.min(30, cc), obsBoxH=Math.min(40, rc);
    var obsOverlay=null, obsInfo=null, obsFrozen=null;
    function ensureObs(){
        if(!obsOverlay){
            gc.style.position='relative';
            obsOverlay=document.createElement('div'); obsOverlay.className='sgy-observe-overlay';
            obsOverlay.style.display='none'; gc.appendChild(obsOverlay);
            obsInfo=document.createElement('div'); obsInfo.className='sgy-observe-info';
            obsInfo.style.display='none'; gc.appendChild(obsInfo);
        }
    }
    function gCS(){var r=gCanvas.getBoundingClientRect();var cx=gCanvas.offsetLeft, cy=gCanvas.offsetTop;return{cssW:r.width/cc, cssH:r.height/rc, left:r.left, top:r.top, cx:cx, cy:cy};}
    function updateObs(cs, mx, my){
        var bx=Math.max(0,Math.min(mx-Math.floor(obsBoxW/2), cc-obsBoxW));
        var by=Math.max(0,Math.min(my-Math.floor(obsBoxH/2), rc-obsBoxH));
        obsOverlay.style.display='block'; obsInfo.style.display='block';
        obsOverlay.style.left=(cs.cx + bx*cs.cssW)+'px'; obsOverlay.style.top=(cs.cy + by*cs.cssH)+'px';
        obsOverlay.style.width=(obsBoxW*cs.cssW)+'px'; obsOverlay.style.height=(obsBoxH*cs.cssH)+'px';
        obsInfo.style.left=(cs.cx + bx*cs.cssW+2)+'px'; obsInfo.style.top=(cs.cy + by*cs.cssH-22)+'px';
        return{bx:bx, by:by};
    }
    gCanvas.addEventListener('mousemove',function(e){
        if(obsFrozen)return; ensureObs();
        var cs=gCS(), mx=Math.floor((e.clientX-cs.left)/cs.cssW), my=Math.floor((e.clientY-cs.top)/cs.cssH);
        if(mx>=0&&mx<cc&&my>=0&&my<rc){
            var pos=updateObs(cs,mx,my);
            obsInfo.textContent='道'+pos.bx+'-'+(pos.bx+obsBoxW)+' 样点'+pos.by+'-'+(pos.by+obsBoxH)+' (点击固定)';
        }
    });
    gCanvas.addEventListener('click',function(e){
        var cs=gCS(), mx=Math.floor((e.clientX-cs.left)/cs.cssW), my=Math.floor((e.clientY-cs.top)/cs.cssH);
        if(mx<0||mx>=cc||my<0||my>=rc)return;
        ensureObs();
        if(obsFrozen){
            obsFrozen=null; obsOverlay.style.borderColor='#f59e0b'; obsOverlay.style.background='rgba(245,158,11,0.08)';
            var pos=updateObs(cs,mx,my);
            obsInfo.textContent='道'+pos.bx+'-'+(pos.bx+obsBoxW)+' 样点'+pos.by+'-'+(pos.by+obsBoxH)+' (点击固定)';
        }else{
            var pos=updateObs(cs,mx,my);
            obsFrozen={traceStart:pos.bx, traceEnd:pos.bx+obsBoxW, sampleStart:pos.by, sampleEnd:pos.by+obsBoxH};
            obsOverlay.style.borderColor='#ec4899'; obsOverlay.style.background='rgba(236,72,153,0.1)';
            obsInfo.textContent='道'+pos.bx+'-'+(pos.bx+obsBoxW)+' 样点'+pos.by+'-'+(pos.by+obsBoxH)+' (已固定，点击解除)';
            fetchLocalBitStats(obsFrozen);
        }
    });
    gCanvas.addEventListener('mouseleave',function(){if(obsOverlay&&!obsFrozen){obsOverlay.style.display='none';obsInfo.style.display='none';}});
}

function getErr(e){return e&&e.response&&e.response.data&&e.response.data.detail?e.response.data.detail:(e.message||'未知错误');}

// ===== Waveform =====
function drawWaveform(){
    if(!heatmapData||heatmapData.length===0){alert('请先加载热力图');return;}
    var tIdx=parseInt($('traceIdx').value)||0;if(tIdx<0||tIdx>=heatmapData.length){alert('道索引超范围');return;}
    var td=heatmapData[tIdx];if(!td||td.length===0)return;var sc=td.length;var c=$('waveformContainer');c.innerHTML='';
    var w=900,h=320,m={top:15,right:20,bottom:35,left:55},pw=w-m.left-m.right,ph=h-m.top-m.bottom;
    var canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;canvas.style.cssText='display:block;margin:0 auto;';c.appendChild(canvas);
    var ctx=canvas.getContext('2d'),tMin=Infinity,tMax=-Infinity;
    for(var i=0;i<sc;i++){if(td[i]<tMin)tMin=td[i];if(td[i]>tMax)tMax=td[i];}
    var tR=tMax-tMin||1,pad=tR*0.05;tMin-=pad;tMax+=pad;tR=tMax-tMin;
    ctx.translate(m.left,m.top);
    ctx.strokeStyle='#e8e8ec';ctx.lineWidth=0.5;
    for(var g=0;g<=4;g++){var gy=Math.round(g*ph/4)+0.5;ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(pw,gy);ctx.stroke();}
    for(var g=0;g<=8;g++){var gx=Math.round(g*pw/8)+0.5;ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,ph);ctx.stroke();}
    var zeroY=tMin<=0&&tMax>=0?Math.round((1-(0-tMin)/tR)*ph)+0.5:-1;
    if(zeroY>=0&&zeroY<=ph){ctx.strokeStyle='#ccc';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,zeroY);ctx.lineTo(pw,zeroY);ctx.stroke();}
    ctx.strokeStyle='#5b6af0';ctx.lineWidth=1;ctx.beginPath();var fp=true;
    for(var i=0;i<sc;i++){var x=(i/(sc-1))*pw,y=(1-(td[i]-tMin)/tR)*ph;if(fp){ctx.moveTo(x,y);fp=false;}else ctx.lineTo(x,y);}
    ctx.stroke();
    ctx.globalAlpha=0.12;ctx.lineTo(pw,ph);ctx.lineTo(0,ph);ctx.closePath();ctx.fillStyle='#5b6af0';ctx.fill();ctx.globalAlpha=1;
    ctx.fillStyle='#999';ctx.font='10px Inter';ctx.textAlign='center';ctx.fillText('采样点',pw/2,ph+28);
    ctx.save();ctx.translate(-m.left+14,ph/2);ctx.rotate(-Math.PI/2);ctx.fillText('振幅',0,0);ctx.restore();
    ctx.fillStyle='#999';ctx.textAlign='right';
    for(var g=0;g<=4;g++){var val=tMax-(g/4)*tR,gy=Math.round(g*ph/4);ctx.fillText(val.toFixed(val<0.01?4:2),m.left-4,gy+4);}
    $('waveformDesc').textContent='道 #'+tIdx+' 振幅曲线 ('+sc+' 采样点)';
    $('waveformInfo').textContent='道 #'+tIdx+' | '+sc+' 采样 | 值域 ['+tMin.toFixed(2)+', '+tMax.toFixed(2)+']';
}

// ===== Bit Stats =====
var bitStatsData=null,bitStatsTab='exponent';

function runBitStats(){
    if(!sgyMeta||!sgyMeta.stored_path){alert('请先上传并解析 SEG-Y 文件');return;}
    $('bitStatsContainer').innerHTML='<div class="loading-spinner" style="margin:auto;"></div>';
    $('bitStatStatus').textContent='全文件扫描中...';
    axios.get(API_BASE_URL+'/api/sgy/bit-stats',{params:{file_path:sgyMeta.stored_path}}).then(function(resp){
        bitStatsData=resp.data;$('bitStatStatus').textContent='统计完成 | '+bitStatsData.total_samples.toLocaleString()+' 样本';showBsTab(bitStatsTab);
    }).catch(function(err){$('bitStatsContainer').innerHTML='<div class="chart-placeholder">统计失败: '+getErr(err)+'</div>';$('bitStatStatus').textContent='统计失败';});
}
function switchBitTab(tab){bitStatsTab=tab;
    var te=$('tabExponent'),ts=$('tabSign'),tm=$('tabMantissa');
    if(te)te.className='gs-tab-btn'+(tab==='exponent'?' active':'');
    if(ts)ts.className='gs-tab-btn'+(tab==='sign'?' active':'');
    if(tm)tm.className='gs-tab-btn'+(tab==='mantissa'?' active':'');
    if(bitStatsData)showBsTab(tab);
}
function showBsTab(tab){var c=$('bitStatsContainer');c.innerHTML='';if(!bitStatsData)return;
    if(tab==='exponent')_expTab(c,bitStatsData);else if(tab==='sign')_signTab(c,bitStatsData);else _mantTab(c);
}

// Exponent
function _expTab(c,d){var e=d.exponent;
    var cd=document.createElement('div');cd.className='sv-stats-card';
    cd.innerHTML='<div class="sv-stat-box"><span class="sv-stat-val">'+e.entropy.toFixed(4)+'</span><span class="sv-stat-lbl">经验熵 (bits)</span></div><div class="sv-stat-box"><span class="sv-stat-val">'+e.theoretical_lower_bound.toFixed(4)+'</span><span class="sv-stat-lbl">理论下界 (bits)</span></div><div class="sv-stat-box"><span class="sv-stat-val">'+(e.non_zero_ratio*100).toFixed(2)+'%</span><span class="sv-stat-lbl">非零指数比例</span></div><div class="sv-stat-box"><span class="sv-stat-val">'+d.total_samples.toLocaleString()+'</span><span class="sv-stat-lbl">总采样点数</span></div>';
    c.appendChild(cd);
    var hd=document.createElement('div');hd.style.cssText='margin-top:var(--spacing-md);';c.appendChild(hd);_hist(hd,e.histogram);
    var hdr=document.createElement('div');hdr.style.cssText='margin-top:var(--spacing-md);text-align:center;font-size:1rem;font-weight:600;color:var(--text-primary);';hdr.textContent='指数均值空间分布 (Inline × Crossline)';c.appendChild(hdr);
    var hq=document.createElement('div');hq.style.cssText='display:flex;justify-content:center;margin-top:var(--spacing-sm);';c.appendChild(hq);_heat(hq,e.spatial_grid);
}

// Sign
function _signTab(c,d){var s=d.sign;
    var cd=document.createElement('div');cd.className='sv-stats-card';
    cd.innerHTML='<div class="sv-stat-box"><span class="sv-stat-val">'+(s.positive_ratio*100).toFixed(2)+'%</span><span class="sv-stat-lbl">正数比例</span></div><div class="sv-stat-box"><span class="sv-stat-val">'+(s.negative_ratio*100).toFixed(2)+'%</span><span class="sv-stat-lbl">负数比例</span></div><div class="sv-stat-box"><span class="sv-stat-val">'+(s.zero_ratio*100).toFixed(2)+'%</span><span class="sv-stat-lbl">零值比例</span></div><div class="sv-stat-box"><span class="sv-stat-val">'+d.total_samples.toLocaleString()+'</span><span class="sv-stat-lbl">总采样点数</span></div>';
    c.appendChild(cd);
    var pie=document.createElement('div');pie.style.cssText='margin-top:var(--spacing-md);display:flex;justify-content:center;';c.appendChild(pie);_pie(pie,s);
    var hdr=document.createElement('div');hdr.style.cssText='margin-top:var(--spacing-md);text-align:center;font-size:1rem;font-weight:600;color:var(--text-primary);';hdr.textContent='符号均值空间分布: 蓝=负 白=零 红=正 (Inline × Crossline)';c.appendChild(hdr);
    var sg=[];for(var r=0;r<s.spatial_grid.length;r++){var row=s.spatial_grid[r],sr=[];for(var cc=0;cc<row.length;cc++){var tr=row[cc],t=tr.pos+tr.neg+tr.zero;sr.push(t>0?(tr.pos-tr.neg)/t:0);}sg.push(sr);}
    var hq=document.createElement('div');hq.style.cssText='display:flex;justify-content:center;margin-top:var(--spacing-sm);';c.appendChild(hq);_heat(hq,sg);
}

// Mantissa - Bit-plane entropy
function _mantTab(c){var m=bitStatsData.mantissa;if(!m||!m.per_bit_entropy){c.innerHTML='<div class="chart-placeholder" style="text-align:center;padding:60px;"><p>尾数位数据不可用</p></div>';return;}
    // 指标卡
    var cd=document.createElement('div');cd.className='sv-stats-card';
    cd.innerHTML='<div class="sv-stat-box"><span class="sv-stat-val">'+m.total_entropy.toFixed(4)+'</span><span class="sv-stat-lbl">尾数总熵 (bits)</span></div><div class="sv-stat-box"><span class="sv-stat-val">'+m.theoretical_upper_bound.toFixed(0)+'</span><span class="sv-stat-lbl">理论上界 (bits)</span></div><div class="sv-stat-box"><span class="sv-stat-val">'+(m.entropy_ratio*100).toFixed(2)+'%</span><span class="sv-stat-lbl">熵占比</span></div><div class="sv-stat-box"><span class="sv-stat-val">'+bitStatsData.total_samples.toLocaleString()+'</span><span class="sv-stat-lbl">总采样点数</span></div>';
    c.appendChild(cd);
    // 分组熵占比
    var grp=document.createElement('div');grp.className='sv-stats-card';grp.style.marginTop='var(--spacing-sm)';
    grp.innerHTML='<div class="sv-stat-box"><span class="sv-stat-val">'+m.group_high.toFixed(4)+' ('+(m.group_high_ratio*100).toFixed(1)+'%)</span><span class="sv-stat-lbl">高位 (1-7bit)</span></div><div class="sv-stat-box"><span class="sv-stat-val">'+m.group_mid.toFixed(4)+' ('+(m.group_mid_ratio*100).toFixed(1)+'%)</span><span class="sv-stat-lbl">中位 (8-15bit)</span></div><div class="sv-stat-box"><span class="sv-stat-val">'+m.group_low.toFixed(4)+' ('+(m.group_low_ratio*100).toFixed(1)+'%)</span><span class="sv-stat-lbl">低位 (16-23bit)</span></div>';
    c.appendChild(grp);
    // 位平面图
    var hd=document.createElement('div');hd.style.cssText='margin-top:var(--spacing-md);';c.appendChild(hd);
    _drawBitPlane(hd,m);
}

function _drawBitPlane(div,m){
    var perBit=m.per_bit_entropy, cumEnt=m.cumulative_entropy, n=perBit.length;
    var w=860,h=300,ml=50,mb=35,mt=20,mr=50,pw=w-ml-mr,ph=h-mt-mb;
    var canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;canvas.style.cssText='display:block;margin:0 auto;';div.appendChild(canvas);
    var ctx=canvas.getContext('2d');
    ctx.fillStyle='#333';ctx.font='13px Inter';ctx.textAlign='center';ctx.fillText('尾数位平面熵 (高位→低位, 23 bits)',w/2,15);

    // 背景: 三区域
    var zones=[{s:16,e:22,c:'rgba(91,106,240,0.06)',l:'高位 1-7bit'},
               {s:8,e:15,c:'rgba(16,185,129,0.06)',l:'中位 8-15bit'},
               {s:0,e:7,c:'rgba(245,158,11,0.06)',l:'低位 16-23bit'}];
    ctx.save();ctx.translate(ml,mt);
    zones.forEach(function(z){
        var x1=(z.s/n)*pw,x2=((z.e+1)/n)*pw;ctx.fillStyle=z.c;ctx.fillRect(x1,0,x2-x1,ph);
        ctx.fillStyle='#999';ctx.font='9px Inter';ctx.textAlign='center';ctx.fillText(z.l,x1+(x2-x1)/2,ph+mb-6);
    });

    var maxBar=Math.max.apply(null,perBit)||1;
    var maxCum=cumEnt[n-1]||1;
    var barW=Math.max(2,pw/n-2);

    // 左Y轴: 每bit熵 (柱状图)
    for(var i=0;i<n;i++){
        var bh=(perBit[i]/maxBar)*ph,x=(i/n)*pw;
        ctx.fillStyle='#5b6af0';ctx.fillRect(x+1,ph-bh,barW,bh);
    }
    ctx.fillStyle='#5b6af0';ctx.font='10px Inter';ctx.textAlign='right';
    for(var i=0;i<=4;i++){var y=ph-(i/4)*ph;ctx.fillText((i*maxBar/4).toFixed(2),-6,y+3);}
    ctx.save();ctx.translate(-40,ph/2);ctx.rotate(-Math.PI/2);ctx.fillText('每bit熵 (bits)',0,0);ctx.restore();

    // 右Y轴: 累计熵 (折线)
    ctx.strokeStyle='#ef4444';ctx.lineWidth=1.5;ctx.beginPath();
    for(var i=0;i<n;i++){var x=((i+0.5)/n)*pw,y=ph-(cumEnt[i]/maxCum)*ph;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}
    ctx.stroke();
    // 红点
    ctx.fillStyle='#ef4444';
    for(var i=0;i<n;i++){ctx.beginPath();ctx.arc(((i+0.5)/n)*pw,ph-(cumEnt[i]/maxCum)*ph,i%3===0?3:1.5,0,2*Math.PI);ctx.fill();}
    ctx.fillStyle='#ef4444';ctx.font='10px Inter';ctx.textAlign='left';
    for(var i=0;i<=4;i++){var y=ph-(i/4)*ph;ctx.fillText((i*maxCum/4).toFixed(2),pw+6,y+3);}
    ctx.save();ctx.translate(pw+36,ph/2);ctx.rotate(-Math.PI/2);ctx.fillText('累计熵 (bits)',0,0);ctx.restore();

    // X轴刻度
    ctx.fillStyle='#666';ctx.font='9px Inter';ctx.textAlign='center';
    for(var i=0;i<n;i+=4){ctx.fillText('b'+(22-i),(i/n)*pw+barW/2,ph+12);}

    ctx.restore();
}

// Pie chart
function _pie(div,s){var w=340,h=180,canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;div.appendChild(canvas);var ctx=canvas.getContext('2d');ctx.fillStyle='#333';ctx.font='13px Inter';ctx.textAlign='center';ctx.fillText('符号位分布',w/2,18);var cx=90,cy=105,r=50;var total=s.positive_count+s.negative_count+s.zero_count||1;var angles=[s.positive_count/total*2*Math.PI,s.negative_count/total*2*Math.PI,s.zero_count/total*2*Math.PI];var colors=['#5b6af0','#ef4444','#d1d5db'];var labels=['正数','负数','零值'];var sa=-Math.PI/2;for(var i=0;i<3;i++){if(angles[i]===0)continue;ctx.beginPath();ctx.moveTo(cx,cy);ctx.arc(cx,cy,r,sa,sa+angles[i]);ctx.fillStyle=colors[i];ctx.fill();sa+=angles[i];}for(var i=0;i<3;i++){var lx=170,ly=80+i*22;ctx.fillStyle=colors[i];ctx.fillRect(lx,ly,10,10);ctx.fillStyle='#666';ctx.font='11px Inter';ctx.textAlign='left';var cnt=i===0?s.positive_count:(i===1?s.negative_count:s.zero_count);ctx.fillText(labels[i]+': '+cnt.toLocaleString(),lx+15,ly+10);}}

// Histogram
function _hist(div,hist){var w=860,h=220,m={top:20,right:10,bottom:30,left:45},pw=w-m.left-m.right,ph=h-m.top-m.bottom;var canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;canvas.style.cssText='display:block;margin:0 auto;';div.appendChild(canvas);var ctx=canvas.getContext('2d');ctx.fillStyle='#333';ctx.font='13px Inter';ctx.textAlign='center';ctx.fillText('指数分布直方图 (0–255)',w/2,15);ctx.save();ctx.translate(m.left,m.top);var maxC=Math.max.apply(null,hist)||1,barW=Math.max(1,pw/256);for(var i=0;i<256;i++){var bh=(hist[i]/maxC)*ph;ctx.fillStyle=hist[i]>0?'#5b6af0':'#e8e8ec';ctx.fillRect(i*barW,ph-bh,Math.max(1,barW-1),bh);}ctx.strokeStyle='#ccc';ctx.lineWidth=0.5;ctx.beginPath();ctx.moveTo(0,ph);ctx.lineTo(pw,ph);ctx.stroke();ctx.fillStyle='#999';ctx.font='9px Inter';ctx.textAlign='center';for(var i=0;i<=256;i+=32){ctx.fillText(i,i*barW,ph+16);}ctx.fillText('指数值',pw/2,ph+30);ctx.restore();}

// Standard heatmap (value range)
function _heat(div,grid){var rc=grid.length,cc=grid[0]?grid[0].length:0;if(rc===0||cc===0)return;var cW=Math.max(2,Math.floor(600/cc)),cH=Math.max(2,Math.floor(400/rc)),cw=cc*cW,ch=rc*cH;var canvas=document.createElement('canvas');canvas.width=cw+50;canvas.height=ch+30;canvas.style.cssText='display:block;';div.appendChild(canvas);var ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;var vMin=Infinity,vMax=-Infinity;for(var r=0;r<rc;r++){var row=grid[r]||[];for(var c=0;c<row.length;c++){if(row[c]<vMin)vMin=row[c];if(row[c]>vMax)vMax=row[c];}}var vR=vMax-vMin||1;
    function hcF(t){t=Math.max(0,Math.min(1,t));var r,g,b;if(t<0.5){var s=t*2;r=Math.round(33+s*222);g=Math.round(62+s*193);b=Math.round(181+s*74);}else{var s=(t-0.5)*2;r=255;g=Math.round(255-s*220);b=Math.round(255-s*220);}return'rgb('+r+','+g+','+b+')';}
    for(var r=0;r<rc;r++){var row=grid[r]||[];for(var c=0;c<row.length;c++){ctx.fillStyle=hcF((row[c]-vMin)/vR);ctx.fillRect(c*cW,r*cH,cW,cH);}}
    var lw=12,lh=Math.min(ch,200),lC=document.createElement('canvas');lC.width=lw+30;lC.height=lh;lC.style.cssText='display:block;';var lx=lC.getContext('2d');for(var s=0;s<lh;s++){lx.fillStyle=hcF(1-s/lh);lx.fillRect(0,s,lw,1);}lx.fillStyle='#666';lx.font='9px Inter';lx.fillText(vMax.toFixed(2),lw+3,10);lx.fillText(vMin.toFixed(2),lw+3,lh-2);div.appendChild(lC);}

// Sign heatmap (fixed domain -1..1, same style as exponent)
function _heatSign(div,grid){var rc=grid.length,cc=grid[0]?grid[0].length:0;if(rc===0||cc===0)return;var cW=Math.max(2,Math.floor(600/cc)),cH=Math.max(2,Math.floor(400/rc)),cw=cc*cW,ch=rc*cH;var canvas=document.createElement('canvas');canvas.width=cw+55;canvas.height=ch+30;canvas.style.cssText='display:block;';div.appendChild(canvas);var ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;
    function hcF(t){t=Math.max(0,Math.min(1,t));var r,g,b;if(t<0.5){var s=t*2;r=Math.round(33+s*222);g=Math.round(62+s*193);b=Math.round(181+s*74);}else{var s=(t-0.5)*2;r=255;g=Math.round(255-s*220);b=Math.round(255-s*220);}return'rgb('+r+','+g+','+b+')';}
    for(var r=0;r<rc;r++){var row=grid[r]||[];for(var c=0;c<row.length;c++){var t=(row[c]+1)/2;ctx.fillStyle=hcF(t);ctx.fillRect(c*cW,r*cH,cW,cH);}}
    var lw=12,lh=Math.min(ch,200),lC=document.createElement('canvas');lC.width=lw+40;lC.height=lh;lC.style.cssText='display:block;';var lx=lC.getContext('2d');for(var s=0;s<lh;s++){lx.fillStyle=hcF(1-s/lh);lx.fillRect(0,s,lw,1);}lx.fillStyle='#666';lx.font='9px Inter';lx.fillText('+1 正',lw+3,10);lx.fillText('0',lw+3,lh/2+3);lx.fillText('-1 负',lw+3,lh-2);div.appendChild(lC);}

// ===== Local Bit Stats =====
async function fetchLocalBitStats(obs){
    if(!sgyMeta||!sgyMeta.stored_path)return;
    var pIdx=parseInt($('profileIdx').value)||0;
    try{
        var resp=await axios.get(API_BASE_URL+'/api/sgy/local-bit-stats',{
            params:{
                file_path:sgyMeta.stored_path, inline_idx:pIdx,
                trace_start:obs.traceStart, trace_end:obs.traceEnd,
                sample_start:obs.sampleStart, sample_end:obs.sampleEnd
            }
        });
        var d=resp.data;
        renderLocalZoom(d);
        renderLocalSign(d);
        renderLocalExp(d);
        renderLocalMant(d);
        renderLocalMantBits(d);
    }catch(e){console.error('local bit stats:',e);}
}

function _viridis(t){t=Math.max(0,Math.min(1,t));var c;if(t<0.25){var s=t/0.25;c=[68+s,1+127*s,84+89*s];}else if(t<0.5){var s=(t-0.25)/0.25;c=[69,128+63*s,173-35*s];}else if(t<0.75){var s=(t-0.5)/0.25;c=[69+121*s,191+32*s,138-55*s];}else{var s=(t-0.75)/0.25;c=[190+63*s,223+8*s,83-46*s];}return 'rgb('+Math.round(c[0])+','+Math.round(c[1])+','+Math.round(c[2])+')';}
function renderLocalZoom(d){
    var grid=d.raw_grid, nTraces=grid.length, nSamples=grid[0]?grid[0].length:0;
    if(nTraces===0||nSamples===0)return;
    var cW=Math.max(3,Math.min(20,Math.floor(220/nTraces))), cH=Math.max(3,Math.min(20,Math.floor(180/nSamples)));
    var w=nTraces*cW, h=nSamples*cH;
    var cv=$('localZoomCanvas'); cv.width=w; cv.height=h; cv.style.width=w+'px'; cv.style.height=h+'px';
    var ctx=cv.getContext('2d'); ctx.imageSmoothingEnabled=false;
    var vMin=Infinity,vMax=-Infinity;
    for(var t=0;t<nTraces;t++){var row=grid[t]||[];for(var s=0;s<row.length;s++){if(row[s]<vMin)vMin=row[s];if(row[s]>vMax)vMax=row[s];}}
    var lo=vMin+0.02*(vMax-vMin), hi=vMax-0.02*(vMax-vMin), vR=hi-lo||1;
    for(var si=0;si<nSamples;si++){for(var t=0;t<nTraces;t++){var row=grid[t]||[];var gs=Math.round(Math.max(0,Math.min(1,(row[si]-lo)/vR))*255);ctx.fillStyle='rgb('+gs+','+gs+','+gs+')';ctx.fillRect(t*cW,si*cH,cW,cH);}}
}

function renderLocalSign(d){
    var s=d.sign, grid=s.grid, nTraces=grid.length, nSamples=grid[0]?grid[0].length:0;
    if(nTraces===0||nSamples===0)return;
    var cW=Math.max(3,Math.min(20,Math.floor(220/nTraces))), cH=Math.max(3,Math.min(20,Math.floor(180/nSamples)));
    var w=nTraces*cW, h=nSamples*cH;
    var cv=$('localSignCanvas'); cv.width=w; cv.height=h; cv.style.width=w+'px'; cv.style.height=h+'px';
    var ctx=cv.getContext('2d'); ctx.imageSmoothingEnabled=false;
    for(var si=0;si<nSamples;si++){for(var t=0;t<nTraces;t++){var row=grid[t]||[];var v=row[si]; ctx.fillStyle=v===1?'#0B79B4':'#F4F4F2'; ctx.fillRect(t*cW,si*cH,cW,cH);}}
    ctx.fillStyle='rgba(255,255,255,.92)';ctx.strokeStyle='#B0B8C0';ctx.lineWidth=0.6;
    ctx.beginPath();ctx.roundRect(w-62,h-36,58,32,4);ctx.fill();ctx.stroke();
    ctx.fillStyle='#F4F4F2';ctx.strokeStyle='#30363D';ctx.lineWidth=0.4;ctx.fillRect(w-56,h-27,12,10);ctx.strokeRect(w-56,h-27,12,10);
    ctx.fillStyle='#0B79B4';ctx.fillRect(w-56,h-14,12,10);ctx.strokeRect(w-56,h-14,12,10);
    ctx.fillStyle='#374151';ctx.font='9px Inter';ctx.textAlign='left';ctx.fillText('0',w-40,h-19);ctx.fillText('1',w-40,h-5);
}

function renderLocalExp(d){
    var e=d.exponent, grid=e.grid, raw=d.raw_grid, nTraces=grid.length, nSamples=grid[0]?grid[0].length:0;
    if(nTraces===0||nSamples===0)return;
    var cW=Math.max(3,Math.min(20,Math.floor(220/nTraces))), cH=Math.max(3,Math.min(20,Math.floor(180/nSamples)));
    var w=nTraces*cW, h=nSamples*cH;
    var cv=$('localExpCanvas'); cv.width=w; cv.height=h; cv.style.width=w+'px'; cv.style.height=h+'px';
    var ctx=cv.getContext('2d'); ctx.imageSmoothingEnabled=false;
    var vMin=Infinity,vMax=-Infinity;
    for(var t=0;t<nTraces;t++){var row=grid[t]||[];for(var s=0;s<row.length;s++){if(row[s]<vMin)vMin=row[s];if(row[s]>vMax)vMax=row[s];}}
    var vR=vMax-vMin||1;
    function drawExp(){ctx.clearRect(0,0,w,h);for(var si=0;si<nSamples;si++){for(var t=0;t<nTraces;t++){var row=grid[t]||[];ctx.fillStyle=_viridis((row[si]-vMin)/vR);ctx.fillRect(t*cW,si*cH,cW,cH);}}}
    drawExp();
    var lines=[{x:Math.round(nTraces*0.25), color:'#E53935'},{x:Math.round(nTraces*0.5), color:'#00B7FF'},{x:Math.round(nTraces*0.75), color:'#FF8F00'}];
    var dragging=-1;
    function redrawLines(){drawExp();lines.forEach(function(l){var px=l.x*cW+0.5;ctx.strokeStyle=l.color;ctx.lineWidth=1.4;ctx.setLineDash([5,3]);ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px,h);ctx.stroke();});ctx.fillStyle='rgba(255,255,255,.92)';ctx.strokeStyle='#B0B8C0';ctx.lineWidth=0.6;ctx.setLineDash([]);ctx.beginPath();ctx.roundRect(w-62,h-54,58,50,4);ctx.fill();ctx.stroke();lines.forEach(function(l,i){ctx.strokeStyle=l.color;ctx.lineWidth=1.6;ctx.setLineDash([5,3]);ctx.beginPath();ctx.moveTo(w-56,h-44+i*16);ctx.lineTo(w-36,h-44+i*16);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#374151';ctx.font='9px Inter';ctx.textAlign='left';ctx.fillText('L'+(i+1),w-56,h-35+i*16);});}
    redrawLines();cv.style.cursor='ew-resize';
    cv.addEventListener('mousedown',function(e){var rect=cv.getBoundingClientRect(),mx=Math.floor((e.clientX-rect.left)/cW);for(var i=0;i<lines.length;i++){if(Math.abs(mx-lines[i].x)<=1){dragging=i;cv.style.cursor='grabbing';break;}}});
    cv.addEventListener('mousemove',function(e){if(dragging<0){var rect=cv.getBoundingClientRect(),mx=Math.floor((e.clientX-rect.left)/cW),over=false;for(var i=0;i<lines.length;i++){if(Math.abs(mx-lines[i].x)<=1){over=true;break;}}cv.style.cursor=over?'col-resize':'ew-resize';return;}var rect=cv.getBoundingClientRect(),mx=Math.floor((e.clientX-rect.left)/cW);lines[dragging].x=Math.max(0,Math.min(nTraces-1,mx));redrawLines();renderAmpCurves(d,lines,nTraces,nSamples);});
    document.addEventListener('mouseup',function(){dragging=-1;cv.style.cursor='ew-resize';});
    renderAmpCurves(d,lines,nTraces,nSamples);
}
function renderAmpCurves(d, lines, nTraces, nSamples){
    var raw=d.raw_grid; $('localAmpPanel').style.display='block';
    var aw=340, ah=180, cv=$('localAmpCanvas'); cv.width=aw; cv.height=ah; cv.style.width=aw+'px'; cv.style.height=ah+'px';
    var ctx=cv.getContext('2d'); ctx.clearRect(0,0,aw,ah);
    var m={top:15,right:15,bottom:28,left:35}, pw=aw-m.left-m.right, ph=ah-m.top-m.bottom;
    var gMin=Infinity,gMax=-Infinity;
    lines.forEach(function(l){var ti=l.x; if(ti<0||ti>=nTraces)return; var row=raw[ti]||[]; for(var s=0;s<row.length;s++){if(row[s]<gMin)gMin=row[s];if(row[s]>gMax)gMax=row[s];}});
    if(!isFinite(gMin))return;
    var gR=gMax-gMin||1, pad=gR*0.05; gMin-=pad; gMax+=pad; gR=gMax-gMin;
    ctx.save();ctx.translate(m.left,m.top);
    ctx.strokeStyle='#dde';ctx.lineWidth=0.5;ctx.setLineDash([4,4]);
    for(var g=0;g<=4;g++){var gy=Math.round(g*ph/4)+0.5;ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(pw,gy);ctx.stroke();}
    ctx.setLineDash([]);
    ctx.fillStyle='#6b7280';ctx.font='9px Inter';ctx.textAlign='right';
    for(g=0;g<=4;g++){ctx.fillText((gMax-(g/4)*gR).toFixed(1),-4,Math.round(g*ph/4)+4);}
    ctx.fillText('振幅',-4,ph/2-8);
    lines.forEach(function(l,i){
        var ti=l.x; if(ti<0||ti>=nTraces)return; var row=raw[ti]||[]; if(row.length===0)return;
        ctx.strokeStyle=l.color;ctx.lineWidth=1.5;ctx.beginPath();
        for(var s=0;s<row.length;s++){var x=(s/(row.length-1))*pw, y=(1-(row[s]-gMin)/gR)*ph; if(s===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);}
        ctx.stroke();
    });
    ctx.fillStyle='#6b7280';ctx.font='9px Inter';ctx.textAlign='left';
    lines.forEach(function(l,i){ctx.fillStyle=l.color;ctx.fillRect(2,ph+10+i*12,8,8);ctx.fillStyle='#374151';ctx.fillText('L'+(i+1),13,ph+17+i*12);});
    ctx.fillStyle='#6b7280';ctx.font='9px Inter';ctx.textAlign='center';ctx.fillText('样点 →', pw/2, ph+12);
    ctx.restore();
}

function renderLocalMant(d){
    var m=d.mantissa, grid=m.high_byte_grid, nTraces=grid.length, nSamples=grid[0]?grid[0].length:0;
    if(nTraces===0||nSamples===0)return;
    var cW=Math.max(3,Math.min(20,Math.floor(220/nTraces))), cH=Math.max(3,Math.min(20,Math.floor(180/nSamples)));
    var w=nTraces*cW, h=nSamples*cH;
    var cv=$('localMantCanvas'); cv.width=w; cv.height=h; cv.style.width=w+'px'; cv.style.height=h+'px';
    var ctx=cv.getContext('2d'); ctx.imageSmoothingEnabled=false;
    var vMin=Infinity,vMax=-Infinity;
    for(var t=0;t<nTraces;t++){var row=grid[t]||[];for(var s=0;s<row.length;s++){if(row[s]<vMin)vMin=row[s];if(row[s]>vMax)vMax=row[s];}}
    var vR=vMax-vMin||1;
    for(var si=0;si<nSamples;si++){for(var t=0;t<nTraces;t++){var row=grid[t]||[];ctx.fillStyle=_viridis((row[si]-vMin)/vR);ctx.fillRect(t*cW,si*cH,cW,cH);}}
}

function switchLocalView(view){
    $('localOverview').style.display = view==='overview'?'block':'none';
    $('localMantBits').style.display = view==='mantbits'?'block':'none';
    $('tabOverview').classList.toggle('active', view==='overview');
    $('tabMantBits').classList.toggle('active', view==='mantbits');
}

function renderLocalMantBits(d){
    var m=d.mantissa, full=m.full_grid, highBits=22, lowBits=7, nRows=full.length, nCols=full[0]?full[0].length:0;
    if(nRows===0||nCols===0)return;
    // Use up to 40 consecutive values (choose a middle trace)
    var traceIdx=Math.floor(nRows/2), displayCount=Math.min(nCols,40);
    var traceData=full[traceIdx]||[];
    var HIGH_BITS=[22,21,20,19,18,17,16,15], LOW_BITS=[7,6,5,4,3,2,1,0];
    var MAN_H=[22,21,20,19], MAN_L=[3,2,1,0];
    var cellS=18, hc=HIGH_BITS.length, lc=LOW_BITS.length;

    // High 8 bits canvas
    var hw=hc*cellS, hh=displayCount*cellS;
    var hcv=$('localMantHighCanvas'); hcv.width=hw; hcv.height=hh; hcv.style.width=hw+'px'; hcv.style.height=hh+'px';
    var hctx=hcv.getContext('2d'); hctx.imageSmoothingEnabled=false;
    for(var ri=0;ri<displayCount;ri++){
        var val=full[traceIdx][ri]||0;
        for(var bi=0;bi<hc;bi++){
            var bit=(val>>HIGH_BITS[bi])&1, isMan=MAN_H.indexOf(HIGH_BITS[bi])>=0;
            hctx.fillStyle=bit?(isMan?'#FCD69A':'#E5E5E5'):'#FFFFFF';
            hctx.fillRect(bi*cellS,ri*cellS,cellS-0.5,cellS-0.5);
        }
    }
    // Grid lines
    hctx.strokeStyle='#000';hctx.lineWidth=0.45;
    for(var gi=0;gi<=hc;gi++){hctx.beginPath();hctx.moveTo(gi*cellS-.5,0);hctx.lineTo(gi*cellS-.5,hh);hctx.stroke();}
    for(var gi=0;gi<=displayCount;gi++){hctx.beginPath();hctx.moveTo(0,gi*cellS-.5);hctx.lineTo(hw,gi*cellS-.5);hctx.stroke();}
    // Column labels
    hctx.fillStyle='#374151';hctx.font='9px Inter';hctx.textAlign='center';
    for(var bi=0;bi<hc;bi++){hctx.fillText('m'+HIGH_BITS[bi], bi*cellS+cellS/2, hh-3);}
    // Torn right edge
    _tornEdge(hctx, hw, hh, 'right');

    // Low 8 bits canvas
    var lw=lc*cellS, lh=displayCount*cellS;
    var lcv=$('localMantLowCanvas'); lcv.width=lw; lcv.height=lh; lcv.style.width=lw+'px'; lcv.style.height=lh+'px';
    var lctx=lcv.getContext('2d'); lctx.imageSmoothingEnabled=false;
    for(var ri=0;ri<displayCount;ri++){
        var val=full[traceIdx][ri]||0;
        for(var bi=0;bi<lc;bi++){
            var bit=(val>>LOW_BITS[bi])&1, isMan=MAN_L.indexOf(LOW_BITS[bi])>=0;
            lctx.fillStyle=bit?(isMan?'#FCD69A':'#E5E5E5'):'#FFFFFF';
            lctx.fillRect(bi*cellS,ri*cellS,cellS-0.5,cellS-0.5);
        }
    }
    lctx.strokeStyle='#000';lctx.lineWidth=0.45;
    for(var gi=0;gi<=lc;gi++){lctx.beginPath();lctx.moveTo(gi*cellS-.5,0);lctx.lineTo(gi*cellS-.5,lh);lctx.stroke();}
    for(var gi=0;gi<=displayCount;gi++){lctx.beginPath();lctx.moveTo(0,gi*cellS-.5);lctx.lineTo(lw,gi*cellS-.5);lctx.stroke();}
    for(var bi=0;bi<lc;bi++){lctx.fillStyle='#374151';lctx.font='9px Inter';lctx.textAlign='center';lctx.fillText('m'+LOW_BITS[bi], bi*cellS+cellS/2, lh-3);}
    _tornEdge(lctx, 0, lh, 'left');

    // Legend canvas
    var lgw=80, lgh=110, lgcv=$('localMantLegend'); lgcv.width=lgw; lgcv.height=lgh; lgcv.style.width=lgw+'px'; lgcv.style.height=lgh+'px';
    var lgctx=lgcv.getContext('2d');
    var items=[{color:'#FFFFFF',label:'0'},{color:'#E5E5E5',label:'other 1'},{color:'#FCD69A',label:'managed 1'}];
    items.forEach(function(item,i){lgctx.fillStyle=item.color;lgctx.fillRect(10,10+i*30,20,20);lgctx.strokeStyle='#000';lgctx.lineWidth=0.5;lgctx.strokeRect(10,10+i*30,20,20);lgctx.fillStyle='#374151';lgctx.font='11px Inter';lgctx.textAlign='left';lgctx.fillText(item.label,36,25+i*30);});
}
function _tornEdge(ctx, edgeX, rows, side){
    var y=[], n=25; for(var i=0;i<n;i++)y.push(-0.5+i*(rows-0.5+0.5)/n);
    var phase=y.map(function(_,i){return i%2;});
    var offset=phase.map(function(p){return 2.5+2.5*p;});
    var jaggedX=side==='right'?offset.map(function(o,i){return edgeX-o;}):offset.map(function(o,i){return edgeX+4-o;});
    ctx.fillStyle='#f8f9fb';ctx.beginPath();
    ctx.moveTo(edgeX,-1);ctx.lineTo(edgeX,rows+1);
    for(var i=n-1;i>=0;i--)ctx.lineTo(jaggedX[i],y[i]);
    ctx.closePath();ctx.fill();
    ctx.strokeStyle='#000';ctx.lineWidth=1.1;ctx.beginPath();
    for(var i=0;i<n;i++){if(i===0)ctx.moveTo(jaggedX[i],y[i]);else ctx.lineTo(jaggedX[i],y[i]);}
    ctx.stroke();
}
