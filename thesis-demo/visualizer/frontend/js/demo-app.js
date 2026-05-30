/**
 * Demo App v2 — Small Volume Pipeline (2×2×100)
 * 分步处理，真实数据，逻辑驱动
 */
(function() {
    'use strict';

    var State = {
        currentStep: 0, isPlaying: false, playInterval: null,
        sampleIndex: 0, speed: 3, sampleCount: 400,
        sgyTaskId: null, sgyFileName: '',
        sgyStats: null, statsCached: false,
        // apiData holds real data fetched from backend
        apiData: {}
    };
    var STEP_TYPES = [false, false, false, false, false];  // all 3D
    var STEP_COUNT = 5;
    var sm, cameraCtrl, stepRenderer, demoAPI;
    var canvas2d, ctx2d, cw, ch;
    var backendOnline = true;
    var pendingFetches = 0;

    // ====== helpers ======
    function hasRealData() { return !!State.sgyTaskId; }
    function hasPredictData() { return State.apiData.predict && State.apiData.predict.probabilities; }
    function hasStats() { return State.sgyStats && State.sgyStats.compressed_size > 0; }

    // ====== Canvas 2D (fallback) ======
    function initCanvas2D() {
        canvas2d = document.getElementById('main2dCanvas'); if (!canvas2d) return;
        ctx2d = canvas2d.getContext('2d'); resizeCanvas2D();
        window.addEventListener('resize', resizeCanvas2D);
    }
    function resizeCanvas2D() {
        if(!canvas2d) return;
        var stg = document.getElementById('canvas2dStage');
        cw = stg.clientWidth; ch = stg.clientHeight;
        if(cw===0||ch===0){
            console.warn('[resizeCanvas2D] zero size detected, using window fallback');
            cw = window.innerWidth; ch = window.innerHeight - 126;
        }
        var dpr = Math.min(window.devicePixelRatio||1, 2);
        canvas2d.width = cw*dpr; canvas2d.height = ch*dpr;
        canvas2d.style.width = cw+'px'; canvas2d.style.height = ch+'px';
        ctx2d.setTransform(dpr,0,0,dpr,0,0);
    }

    // ====== 3D ======
    function init3D() {
        if(!document.getElementById('three-container')) return;
        sm = new SceneManager('three-container');
        cameraCtrl = new CameraController(sm.camera, sm.renderer.domElement);
        sm.cameraController = cameraCtrl;
        stepRenderer = new StepRenderer(sm);
        demoAPI = new DemoAPI();
        stepRenderer.register(0, new Step1BitDecomposer());
        stepRenderer.register(1, new Step2CausalPatch());
        stepRenderer.register(2, new Step3CNNPredict());
        stepRenderer.register(3, new Step4RangeCoding());
        stepRenderer.register(4, new Step6Dashboard());
        sm.addUpdater(function(dt,t){cameraCtrl.update();stepRenderer.update(dt,t);});
        sm.renderer.domElement.addEventListener('pointermove',function(e){WebGLUtils.Tooltip3D.setPos(e.clientX,e.clientY);stepRenderer.onPointerMove(e);});
        sm.renderer.domElement.addEventListener('click',function(e){stepRenderer.onClick(e);});
    }

    function showOfflineBadge(show) { var e=document.getElementById('offlineBadge'); if(e)e.style.display=show?'block':'none'; }
    function updateDataSourceLabel() {
        var e=document.getElementById('dataSourceLabel'); if(!e)return;
        if(hasRealData()){e.textContent='真实 ('+State.sampleCount+'点)';e.style.color='var(--demo-success)';}
        else{e.textContent='无数据';e.style.color='var(--demo-exp)';}
    }

    // ====== Placeholder canvas ======
    function drawPlaceholder(msg) {
        console.log('[drawPlaceholder] msg='+(msg||'default')+' cw='+cw+' ch='+ch);
        if(!ctx2d||!canvas2d){ console.warn('[drawPlaceholder] no ctx2d/canvas2d'); return; }
        resizeCanvas2D();
        ctx2d.clearRect(0,0,cw,ch);
        ctx2d.font = '18px "Segoe UI",sans-serif'; ctx2d.fillStyle = '#e2e8f0'; ctx2d.textAlign = 'center';
        ctx2d.fillText(msg||'请先加载 SEG-Y 文件', cw/2, ch/2-10);
        ctx2d.font = '13px "Segoe UI",sans-serif'; ctx2d.fillStyle = '#94a3b8';
        ctx2d.fillText('点击左侧"加载 SEG-Y 文件"按钮', cw/2, ch/2+20);
    }

    // ====== SEG-Y upload ======
    async function onSGYFileSelected(e) {
        var f=e.target.files[0]; if(!f)return;
        var st=document.getElementById('sgyLoadStatus'), btn=document.getElementById('btnLoadSGY');
        if(st)st.textContent='上传并初始化模型...'; if(btn)btn.textContent='处理中...';
        try {
            var r=await demoAPI.uploadSGY(f);
            if(r&&r.task_id){
                State.sgyTaskId=r.task_id;
                State.sgyFileName=r.filename;
                State.sampleCount=r.sample_count||400;
                State.sgyStats=null; State.statsCached=false; State.apiData={};
                // 更新滑块范围
                var slider=document.getElementById('sampleSlider');
                if(slider){slider.max=State.sampleCount-1; slider.value=0;}
                State.sampleIndex=0;
                document.getElementById('sampleIdxVal').textContent='0';
                if(st)st.textContent='就绪 ('+State.sampleCount+'采样点)';
                console.log('[upload] task_id='+r.task_id+' shape='+JSON.stringify(r.shape));
                gotoStep(State.currentStep);
            }else{
                if(st)st.textContent='上传失败';
            }
        }catch(err){
            console.error('[upload] error:', err);
            if(st)st.textContent='上传失败: '+((err&&err.message)||'');
        }
        if(btn)btn.textContent='加载 SEG-Y 文件';
        updateDataSourceLabel();
    }

    // ====== Coord helper (2×2×100) ======
    function getCoord(){
        var t=State.sampleIndex;
        var s=t%100;
        var tc=Math.floor(t/100)%2;
        var pc=Math.floor(t/200)%2;
        return [pc,tc,s];
    }
    function updateCoordDisplay(){
        var c=getCoord();
        var cp=document.getElementById('coordP'),ct=document.getElementById('coordT'),cs=document.getElementById('coordS');
        if(cp)cp.textContent=c[0]; if(ct)ct.textContent=c[1]; if(cs)cs.textContent=c[2];
    }

    // ====== API Fetch (new pipeline) ======
    async function fetchStepData(idx){
        if(!hasRealData()){ State.apiData={}; return; }
        if(!demoAPI) return;
        var sampleIdx=State.sampleIndex;
        console.log('[fetchStepData] step='+idx+' sampleIndex='+sampleIdx+' taskId='+State.sgyTaskId);
        pendingFetches++;
        try {
            if(idx===0){
                var r=await demoAPI.decompose(State.sgyTaskId, sampleIdx);
                console.log('[fetchStepData] decompose:', r?{sign:r.sign,exp:r.exp_raw}:null);
                if(r&&r.sign!==undefined){
                    State.apiData.step0=r; backendOnline=true; showOfflineBadge(false);
                }else{
                    console.warn('[fetchStepData] decompose invalid:', r);
                }
            }else if(idx===1){
                var r1=await demoAPI.features(State.sgyTaskId, sampleIdx);
                console.log('[fetchStepData] features:', r1?{channels:(r1.channels||[]).length}:null);
                if(r1&&r1.channels){
                    State.apiData.features=r1; backendOnline=true; showOfflineBadge(false);
                }else{
                    console.warn('[fetchStepData] features invalid:', r1);
                }
            }else if(idx===2){
                var r2=await demoAPI.predict(State.sgyTaskId, sampleIdx);
                console.log('[fetchStepData] predict:', r2?{hasProbs:!!r2.probabilities,actual:r2.actual_symbol}:null);
                if(r2&&r2.probabilities){
                    State.apiData.predict=r2; backendOnline=true; showOfflineBadge(false);
                }else{
                    console.warn('[fetchStepData] predict invalid:', r2);
                }
            }else if(idx===3){
                // Step 3 needs both predict (for prob viz) and encode (for range info)
                var r3=await demoAPI.encode(State.sgyTaskId, sampleIdx);
                console.log('[fetchStepData] encode:', r3?{symbol:r3.symbol,range:[r3.range_low,r3.range_high],bits:r3.bits_output}:null);
                if(r3&&r3.symbol!==undefined){
                    State.apiData.encode=r3; backendOnline=true; showOfflineBadge(false);
                }else{
                    console.warn('[fetchStepData] encode invalid:', r3);
                }
                // Also ensure predict data is available for Step4RangeCoding
                if(!State.apiData.predict){
                    var r3p=await demoAPI.predict(State.sgyTaskId, sampleIdx);
                    if(r3p&&r3p.probabilities) State.apiData.predict=r3p;
                }
            }else if(idx===4){
                if(!State.statsCached){
                    var r4=await demoAPI.stats(State.sgyTaskId);
                    console.log('[fetchStepData] stats:', r4?{ratio:r4.compression_ratio,size:r4.compressed_size}:null);
                    if(r4&&r4.compressed_size!==undefined){
                        State.sgyStats=r4; State.statsCached=true;
                    }
                }
            }
        }catch(e){
            console.error('[fetchStepData] error:', e);
            backendOnline=false; showOfflineBadge(true);
        }
        pendingFetches--;
    }

    // ====== 2D Canvas Renderers (fallback) ======
    function render2D_step0(){
        var d=State.apiData.step0;
        if(!d||d.sign===undefined){ drawPlaceholder('等待 Bit 拆解数据...'); return; }
        resizeCanvas2D(); ctx2d.clearRect(0,0,cw,ch);
        var bits=[], colors={sign:'#ef4444',exp:'#f59e0b',mant:'#06b6d4'};
        bits.push({val:d.sign,type:'sign',idx:31});
        var expRaw = d.exp_raw!==undefined?d.exp_raw:(d.exp!==undefined?d.exp:0);
        var mantVal = d.mant!==undefined?d.mant:0;
        for(var i=7;i>=0;i--) bits.push({val:(expRaw>>i)&1,type:'exp',idx:23+i});
        for(var i=22;i>=0;i--) bits.push({val:(mantVal>>i)&1,type:'mant',idx:i});
        var boxSize=Math.min(34,(cw-80)/32),gap=4;
        var startX=(cw-32*(boxSize+gap))/2,startY=ch/2-70;
        ctx2d.fillStyle='rgba(255,255,255,0.08)';ctx2d.beginPath();ctx2d.roundRect(startX-20,startY-20,32*(boxSize+gap)+40,boxSize*1.3+120,12);ctx2d.fill();
        ctx2d.strokeStyle='rgba(255,255,255,0.2)';ctx2d.lineWidth=1;ctx2d.stroke();
        ctx2d.font='bold 14px "Segoe UI",sans-serif';ctx2d.fillStyle='#e2e8f0';ctx2d.textAlign='center';
        ctx2d.fillText('Bit 31 (MSB) ←  → Bit 0 (LSB) — 真实 SEG-Y 数据',cw/2,startY-28);
        bits.forEach(function(b,i){
            var x=startX+i*(boxSize+gap),y=startY;
            ctx2d.shadowColor=colors[b.type];ctx2d.shadowBlur=b.val?15:0;
            ctx2d.fillStyle=b.val?colors[b.type]:'rgba(30,41,59,0.8)';ctx2d.strokeStyle=colors[b.type];ctx2d.lineWidth=2;
            ctx2d.beginPath();ctx2d.roundRect(x,y,boxSize,boxSize*1.3,4);ctx2d.fill();ctx2d.stroke();ctx2d.shadowBlur=0;
            ctx2d.fillStyle=b.val?'#fff':'#94a3b8';ctx2d.font='bold '+(boxSize*0.5)+'px "Courier New",monospace';
            ctx2d.textAlign='center';ctx2d.textBaseline='middle';ctx2d.fillText(b.val,x+boxSize/2,y+boxSize*0.45);
            ctx2d.fillStyle='#cbd5e1';ctx2d.font=(boxSize*0.25)+'px "Courier New",monospace';ctx2d.fillText(b.idx,x+boxSize/2,y+boxSize*1.1);
        });ctx2d.textBaseline='alphabetic';
        ctx2d.font='12px "Segoe UI",sans-serif';ctx2d.textAlign='center';
        ctx2d.fillStyle=colors.sign;ctx2d.fillText('Sign',startX+boxSize/2,startY+boxSize*1.8);
        ctx2d.fillStyle=colors.exp;ctx2d.fillText('Exponent (8 bits)',startX+boxSize+gap+(boxSize*8+gap*7)/2,startY+boxSize*1.8);
        ctx2d.fillStyle=colors.mant;ctx2d.fillText('Mantissa (23 bits)',startX+(boxSize+gap)*9+(boxSize*23+gap*22)/2,startY+boxSize*1.8);
        var valY=startY+boxSize*2.6;
        var value = d.original!==undefined?d.original:(d.value!==undefined?d.value:0);
        ctx2d.font='bold 28px "Courier New",monospace';ctx2d.fillStyle='#f1f5f9';ctx2d.textAlign='center';
        ctx2d.fillText(value.toExponential(6),cw/2,valY);
        ctx2d.font='13px "Segoe UI",sans-serif';ctx2d.fillStyle='#94a3b8';
        ctx2d.fillText('Sign:'+d.sign+'    Exponent:'+expRaw+' (biased)    Mantissa:0x'+mantVal.toString(16).toUpperCase().padStart(6,'0'),cw/2,valY+28);
    }

    function render2D_step2(){
        var data=State.apiData.predict;
        if(!data||!data.probabilities||data.probabilities.length===0){ drawPlaceholder('等待 CNN 预测数据...'); return; }
        resizeCanvas2D(); ctx2d.clearRect(0,0,cw,ch);
        var probs=data.probabilities, actual=data.actual_symbol, cx=cw/2,cy=ch/2+10;
        var chartW=Math.min(700,cw-140),chartH=220,barW=chartW/256;
        var maxProb=Math.max.apply(null,probs),scale=(chartH/2-10)/maxProb;
        ctx2d.font='13px "Segoe UI",sans-serif';ctx2d.fillStyle='#e2e8f0';ctx2d.textAlign='center';
        ctx2d.fillText('Stage4 CNN 输出: 256-class 真实概率分布',cx,cy-chartH/2-30);
        ctx2d.strokeStyle='#334155';ctx2d.beginPath();ctx2d.moveTo(cx-chartW/2,cy+chartH/2);ctx2d.lineTo(cx+chartW/2,cy+chartH/2);ctx2d.stroke();
        for(var i=0;i<256;i++){var bh=probs[i]*scale,x=cx-chartW/2+i*barW,y=cy+chartH/2-bh;var inten=probs[i]/maxProb;ctx2d.fillStyle='rgb('+(6+inten*233|0)+','+(182+inten*15|0)+','+(212-inten*50|0)+')';ctx2d.fillRect(x,y,barW-0.5,bh);if(i===actual){ctx2d.strokeStyle='#ec4899';ctx2d.lineWidth=2;ctx2d.strokeRect(x-1,y-2,barW+1,bh+2);}}
        ctx2d.fillStyle='#94a3b8';ctx2d.font='10px "Courier New",monospace';for(i=0;i<=256;i+=64)ctx2d.fillText(i,cx-chartW/2+i*barW,cy+chartH/2+16);
        ctx2d.fillText('Symbol (0-255)',cx,cy+chartH/2+32);
        var infX=cx+chartW/2+20,infY=cy-50;
        ctx2d.fillStyle='rgba(30,41,59,0.9)';ctx2d.beginPath();ctx2d.roundRect(infX,infY,160,100,8);ctx2d.fill();ctx2d.strokeStyle='rgba(255,255,255,0.15)';ctx2d.stroke();
        ctx2d.textAlign='left';ctx2d.font='11px "Segoe UI",sans-serif';ctx2d.fillStyle='#94a3b8';
        ctx2d.fillText('True Symbol:',infX+12,infY+22);ctx2d.font='bold 18px "Courier New",monospace';ctx2d.fillStyle='#ec4899';ctx2d.fillText(actual,infX+12,infY+46);
        ctx2d.font='11px "Segoe UI",sans-serif';ctx2d.fillStyle='#94a3b8';ctx2d.fillText('Prob:',infX+12,infY+66);
        ctx2d.font='bold 14px "Courier New",monospace';ctx2d.fillStyle='#06b6d4';ctx2d.fillText((probs[actual]*100).toFixed(2)+'%',infX+12,infY+86);
    }

    function render2D_step3(){
        var data=State.apiData.predict;
        var enc=State.apiData.encode;
        if(!data||!data.probabilities||data.probabilities.length===0){ drawPlaceholder('等待 Range Coding 数据...'); return; }
        resizeCanvas2D(); ctx2d.clearRect(0,0,cw,ch);
        var probs=data.probabilities, symbol=enc?enc.symbol:(data.actual_symbol||0);
        var cdf=[],cum=0;
        for(var i=0;i<256;i++){cum+=probs[i];cdf.push(cum);}
        var cx=cw/2,cy=ch/2,barW=Math.min(600,cw-100),barH=40,startY=cy-100;
        ctx2d.font='13px "Segoe UI",sans-serif';ctx2d.fillStyle='#e2e8f0';ctx2d.textAlign='center';
        ctx2d.fillText('Range Coding — 真实编码',cx,startY-20);
        ctx2d.fillStyle='rgba(255,255,255,0.12)';ctx2d.fillRect(cx-barW/2,startY,barW,barH);
        var cumX=cx-barW/2;for(i=0;i<256;i++){var w=probs[i]*barW;ctx2d.fillStyle=i===symbol?'#ec4899':'hsla('+((i/256)*360)+',70%,50%,0.3)';ctx2d.fillRect(cumX,startY,w,barH);cumX+=w;}
        ctx2d.strokeStyle='rgba(255,255,255,0.3)';ctx2d.lineWidth=2;ctx2d.strokeRect(cx-barW/2,startY,barW,barH);
        ctx2d.fillStyle='#94a3b8';ctx2d.font='11px "Courier New",monospace';ctx2d.fillText('0.0',cx-barW/2,startY+barH+18);ctx2d.fillText('1.0',cx+barW/2,startY+barH+18);
        var symStart=cx-barW/2+(cdf[symbol]-probs[symbol])*barW,symW=probs[symbol]*barW;
        ctx2d.strokeStyle='#ec4899';ctx2d.lineWidth=3;ctx2d.beginPath();ctx2d.moveTo(symStart+symW/2,startY-10);ctx2d.lineTo(symStart+symW/2,startY);ctx2d.stroke();
        ctx2d.fillStyle='#ec4899';ctx2d.font='bold 12px "Segoe UI",sans-serif';ctx2d.textAlign='center';ctx2d.fillText('Symbol '+symbol,symStart+symW/2,startY-16);
        // Range info
        if(enc){
            var infoY=startY+80;
            ctx2d.fillStyle='#e2e8f0';ctx2d.font='12px "Segoe UI",sans-serif';ctx2d.textAlign='center';
            ctx2d.fillText('Range: ['+enc.range_low.toFixed(4)+', '+enc.range_high.toFixed(4)+')  Prob: '+(enc.prob*100).toFixed(2)+'%',cx,infoY);
            ctx2d.fillText('Bits so far: '+enc.bits_output+'  Encoded: '+enc.encoded_count+'/'+enc.total_coords,cx,infoY+20);
            if(enc.bits_string) ctx2d.fillText('Output: '+enc.bits_string,cx,infoY+40);
        }
    }

    function render2DStep(idx){
        console.log('[render2DStep] called idx='+idx);
        if(!ctx2d||!canvas2d){ console.warn('[render2DStep] missing canvas context'); return; }
        if(idx===0)render2D_step0();else if(idx===2)render2D_step2();else if(idx===3)render2D_step3();
        else console.warn('[render2DStep] unknown idx='+idx);
    }

    function setMode2D(){document.getElementById('three-container').style.display='none';document.getElementById('canvas2dStage').style.display='flex';if(sm)sm.stop();}
    function setMode3D(){document.getElementById('three-container').style.display='block';document.getElementById('canvas2dStage').style.display='none';if(sm){sm.onResize();sm.start();}}

    // ====== Formulas / Metrics ======
    var formulas = [
        '<span class="comment"># IEEE 754 Float32 拆解</span><br><span class="keyword">def</span> extract_float_components(data):<br>&nbsp;&nbsp;u32 = data.view(np.uint32)<br>&nbsp;&nbsp;signs = (u32 &gt;&gt; 31) &amp; 0x1<br>&nbsp;&nbsp;exps = (u32 &gt;&gt; 23) &amp; 0xFF<br>&nbsp;&nbsp;mants = u32 &amp; 0x7FFFFF',
        '<span class="comment"># diagonal_causal_edge 特征提取</span><br>feature = build_stage4_features(<br>&nbsp;&nbsp;exps, coord,<br>&nbsp;&nbsp;patch_shape=(9, 17),<br>&nbsp;&nbsp;feature_mode=<span class="string">"diagonal_causal_edge"</span><br>)',
        '<span class="comment"># CNN 概率预测</span><br>logits = Small2DCNN(feature_patch)<br>probs = softmax(logits)  <span class="comment"># 256</span>',
        '<span class="comment"># Range Coder</span><br>cdf = probs_to_cdf(probs)<br>encoder.encode_symbol(symbol, cdf)',
        '<span class="comment"># 压缩汇总</span><br>ratio = raw_bytes / total_bytes'
    ];
    var texts = [
        'Float32 由 32 Bit 组成：1 Sign + 8 Exponent + 23 Mantissa。当前展示真实 SEG-Y 采样点的 bit 拆解结果。',
        '在 profile-trace 平面提取 9×17 因果 Patch。只使用已编码空间位置作为上下文。红色高亮为当前采样点位置。',
        'Small2DCNN 接收因果 Patch，输出 256 类概率分布。粉色 beacon 标记真实 Symbol 位置。',
        '将 CNN 概率量化为 CDF，Range Encoder 根据符号的概率区间缩窄编码。展示真实编码区间和累计输出 bits。',
        '压缩结果 = Stage4 压缩指数 + zstd 压缩符号 + zstd 压缩尾数。基于 2×2×100 小体积的真实统计。'
    ];

    // 每一步的输入/处理/输出说明
    var stepFlow = [
        {input:'Float32 采样值', process:'IEEE 754 位拆解', output:'Sign + Exponent + Mantissa'},
        {input:'Exponent 平面', process:'提取 9×17 Causal Patch', output:'6 通道特征 (Values/Masks/Residual)'},
        {input:'特征 Patch', process:'Small2DCNN + Softmax', output:'256 类概率分布 + 预测 Symbol'},
        {input:'概率分布 + Symbol', process:'CDF 量化 + Range 编码', output:'缩窄区间 + 输出 Bits'},
        {input:'全部采样点编码结果', process:'统计压缩大小', output:'压缩比 + 各通道字节数'}
    ];

    function updateInfoPanel(idx){
        var fb=document.getElementById('formulaBox'),it=document.getElementById('infoText');
        if(fb)fb.innerHTML=formulas[idx]; if(it)it.textContent=texts[idx];
    }

    function updateMetrics(idx){
        var mc=document.getElementById('metricsContent');if(!mc)return;
        if(!hasRealData()){
            mc.innerHTML='<div class="metric-card"><div class="metric-label">状态</div><div class="metric-value" style="font-size:13px;color:var(--demo-exp);">请先加载 SEG-Y</div></div>';
            return;
        }
        var flow=stepFlow[idx];
        var html='<div style="margin-bottom:10px;font-size:11px;color:var(--demo-text-muted);">'
            +'<span style="color:var(--demo-accent);">输入:</span> '+flow.input+'<br>'
            +'<span style="color:var(--demo-exp);">处理:</span> '+flow.process+'<br>'
            +'<span style="color:var(--demo-success);">输出:</span> '+flow.output
            +'</div>';

        if(idx===0){
            var d=State.apiData.step0;
            if(!d) html+='<div class="metric-card"><div class="metric-label">加载中...</div></div>';
            else html+='<div class="metric-card" style="border-left-color:var(--demo-sign);"><div class="metric-label">Sign</div><div class="metric-value">'+d.sign+'<span class="metric-unit">bit</span></div></div>'+
                '<div class="metric-card" style="border-left-color:var(--demo-exp);"><div class="metric-label">Exponent</div><div class="metric-value">'+(d.exp_raw||d.exp)+'<span class="metric-unit">biased</span></div></div>'+
                '<div class="metric-card" style="border-left-color:var(--demo-mant);"><div class="metric-label">Mantissa</div><div class="metric-value">0x'+(d.mant||0).toString(16).toUpperCase().padStart(6,'0')+'</div></div>';
        }else if(idx===1){
            var f=State.apiData.features;
            if(!f) html+='<div class="metric-card"><div class="metric-label">加载中...</div></div>';
            else html+='<div class="metric-card"><div class="metric-label">Patch</div><div class="metric-value">'+f.patch_shape.join('×')+'</div></div>'+
                '<div class="metric-card"><div class="metric-label">通道数</div><div class="metric-value">'+(f.channels?f.channels.length:0)+'</div></div>'+
                '<div class="metric-card"><div class="metric-label">Target Symbol</div><div class="metric-value">'+f.target_symbol+'</div></div>';
        }else if(idx===2||idx===3){
            var p=State.apiData.predict;
            var e=State.apiData.encode;
            if(!p) html+='<div class="metric-card"><div class="metric-label">加载中...</div></div>';
            else{
                html+='<div class="metric-card"><div class="metric-label">真实 Symbol</div><div class="metric-value">'+p.actual_symbol+'</div></div>'+
                    '<div class="metric-card"><div class="metric-label">熵 H</div><div class="metric-value">'+(p.entropy||0).toFixed(3)+'<span class="metric-unit">bits</span></div></div>';
                if(e){
                    html+='<div class="metric-card" style="border-left-color:var(--demo-exp);"><div class="metric-label">Range</div><div class="metric-value">['+e.range_low.toFixed(3)+', '+e.range_high.toFixed(3)+')</div></div>'+
                        '<div class="metric-card" style="border-left-color:var(--demo-mant);"><div class="metric-label">Bits</div><div class="metric-value">'+e.bits_output+'</div></div>';
                }
            }
        }else if(idx===4){
            var s=State.sgyStats;
            if(s&&s.compressed_size){
                var ratio=s.compression_ratio||(s.original_size/(s.compressed_size||1));
                var origMB=(s.original_size/1024/1024).toFixed(1),compKB=(s.compressed_size/1024).toFixed(1);
                html+='<div class="metric-card" style="border-left-color:var(--demo-success);"><div class="metric-label">压缩比</div><div class="metric-value">'+ratio.toFixed(2)+'<span class="metric-unit">:1</span></div></div>'+
                    '<div class="metric-card"><div class="metric-label">原始</div><div class="metric-value">'+origMB+'<span class="metric-unit">MB</span></div></div>'+
                    '<div class="metric-card"><div class="metric-label">压缩后</div><div class="metric-value">'+compKB+'<span class="metric-unit">KB</span></div></div>'+
                    '<div class="metric-card" style="border-left-color:var(--demo-exp);"><div class="metric-label">Exponent</div><div class="metric-value">'+((s.exponent_bytes||0)/1024).toFixed(1)+'<span class="metric-unit">KB</span></div></div>'+
                    '<div class="metric-card" style="border-left-color:var(--demo-sign);"><div class="metric-label">Sign</div><div class="metric-value">'+((s.sign_bytes||0)/1024).toFixed(1)+'<span class="metric-unit">KB</span></div></div>'+
                    '<div class="metric-card" style="border-left-color:var(--demo-mant);"><div class="metric-label">Mantissa</div><div class="metric-value">'+((s.mant_bytes||0)/1024).toFixed(1)+'<span class="metric-unit">KB</span></div></div>';
            }else{
                html+='<div class="metric-card"><div class="metric-label">状态</div><div class="metric-value" style="font-size:13px;">统计计算中...</div></div>';
            }
        }
        mc.innerHTML=html;
    }

    // ====== Navigation ======
    var transitioning=false;
    async function gotoStep(idx){
        console.log('[gotoStep] start idx='+idx+' transitioning='+transitioning);
        if(transitioning||idx<0||idx>=STEP_COUNT){ console.log('[gotoStep] early return'); return; }
        transitioning=true; State.currentStep=idx;
        document.querySelectorAll('.demo-step-tab').forEach(function(t,i){t.classList.toggle('active',i===idx);});
        var pb=document.getElementById('prevStepBtn'),nb=document.getElementById('nextStepBtn'),lb=document.getElementById('stepLabel');
        if(pb)pb.disabled=idx===0; if(nb)nb.disabled=idx===STEP_COUNT-1; if(lb)lb.textContent='步骤 '+(idx+1)+' / '+STEP_COUNT;
        updateInfoPanel(idx); updateDataSourceLabel();
        // Fetch real data
        await fetchStepData(idx);
        updateMetrics(idx);
        if(STEP_TYPES[idx]){
            console.log('[gotoStep] 2D mode idx='+idx);
            if(stepRenderer)await stepRenderer.cleanup(); setMode2D(); render2DStep(idx);
        }else{
            console.log('[gotoStep] 3D mode idx='+idx);
            setMode3D();
            var stepData = { sampleIndex: State.sampleIndex, stats: State.sgyStats };
            if (idx === 0) stepData.apiData = State.apiData.step0;
            else if (idx === 1) stepData.apiData = State.apiData.features;
            else if (idx === 2 || idx === 3){
                // Step 3 needs both predict and encode data
                stepData.apiData = State.apiData.predict || {};
                if(State.apiData.encode){
                    stepData.apiData = Object.assign({}, State.apiData.predict || {}, {encode: State.apiData.encode});
                }
            }
            await stepRenderer.goto(idx, stepData);
        }
        console.log('[gotoStep] done idx='+idx);
        transitioning=false;
    }

    // ====== Playback ======
    function refreshCurrentStep(){
        var idx=State.currentStep;
        console.log('[refreshCurrentStep] idx='+idx+' hasRealData='+hasRealData());
        if(!hasRealData()){
            updateCoordDisplay(); updateMetrics(idx);
            return;
        }
        fetchStepData(idx).then(function(){
            console.log('[refreshCurrentStep] fetch done, rendering idx='+idx);
            updateCoordDisplay(); updateMetrics(idx);
            var slider=document.getElementById('sampleSlider'), val=document.getElementById('sampleIdxVal');
            if(slider)slider.value=State.sampleIndex;
            if(val)val.textContent=State.sampleIndex;
            var stepData = { sampleIndex: State.sampleIndex, stats: State.sgyStats };
            if (idx === 0) stepData.apiData = State.apiData.step0;
            else if (idx === 1) stepData.apiData = State.apiData.features;
            else if (idx === 2 || idx === 3){
                stepData.apiData = State.apiData.predict || {};
                if(State.apiData.encode) stepData.apiData = Object.assign({}, stepData.apiData, {encode: State.apiData.encode});
            }
            stepRenderer.goto(idx, stepData);
        }).catch(function(e){ console.error('[refreshCurrentStep] fetch failed:', e); });
    }
    function startPlayback(){
        if(State.isPlaying||!hasRealData())return;
        State.isPlaying=true;updateStatus('running');
        State.playInterval=setInterval(function(){
            State.sampleIndex=(State.sampleIndex+1)%State.sampleCount;
            refreshCurrentStep();
        },Math.max(200,500-State.speed*50));
    }
    function stopPlayback(){State.isPlaying=false;clearInterval(State.playInterval);updateStatus('paused');}
    function updateStatus(m){var b=document.getElementById('statusBadge');if(b){b.className='status-badge status-'+m;b.textContent=m==='running'?'播放中':'已暂停';}}

    // ====== Events ======
    function bindEvents(){
        document.querySelectorAll('.demo-step-tab').forEach(function(t){t.addEventListener('click',function(){gotoStep(parseInt(t.dataset.step));});});
        document.getElementById('prevStepBtn').addEventListener('click',function(){gotoStep(State.currentStep-1);});
        document.getElementById('nextStepBtn').addEventListener('click',function(){gotoStep(State.currentStep+1);});
        document.addEventListener('keydown',function(e){if(e.key==='ArrowRight')gotoStep(State.currentStep+1);if(e.key==='ArrowLeft')gotoStep(State.currentStep-1);if(e.key===' '){e.preventDefault();State.isPlaying?stopPlayback():startPlayback();}});
        document.getElementById('btnPlay').addEventListener('click',startPlayback);
        document.getElementById('btnPause').addEventListener('click',stopPlayback);
        document.getElementById('btnLoadSGY').addEventListener('click',function(){document.getElementById('sgyFileInput').click();});
        document.getElementById('sgyFileInput').addEventListener('change',onSGYFileSelected);
        var sl=document.getElementById('sampleSlider'),sp=document.getElementById('speedSlider');
        if(sl){sl.addEventListener('input',function(e){State.sampleIndex=parseInt(e.target.value);document.getElementById('sampleIdxVal').textContent=State.sampleIndex;updateCoordDisplay();updateMetrics(State.currentStep);});
               sl.addEventListener('change',function(){refreshCurrentStep();});}
        if(sp)sp.addEventListener('input',function(e){State.speed=parseInt(e.target.value);document.getElementById('speedVal').textContent=State.speed;if(State.isPlaying){stopPlayback();startPlayback();}});
    }

    // ====== Boot ======
    document.addEventListener('DOMContentLoaded',function(){
        setTimeout(function(){
            try{initCanvas2D();init3D();bindEvents();gotoStep(0).then(function(){var ld=document.getElementById('loading3d');if(ld)ld.classList.add('hidden');}).catch(function(err){console.error(err);var ld=document.getElementById('loading3d');if(ld)ld.innerHTML='<div style="color:#ef4444">初始化失败</div>';});}
            catch(err){console.error(err);var ld=document.getElementById('loading3d');if(ld)ld.innerHTML='<div style="color:#ef4444">引擎失败</div>';}
        },300);
    });
})();
