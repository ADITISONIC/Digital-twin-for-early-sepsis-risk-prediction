import { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, AlertTriangle, Brain, HeartPulse, Pause, Play, RotateCcw, ShieldCheck, Thermometer, Wind, Database, Radio, Siren, CheckCircle2 } from 'lucide-react';
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api, Prediction } from './api';
import './styles.css';

type Mode = 'dataset' | 'icu';
type Row = Record<string, any>;

const clamp = (x:number,a:number,b:number)=>Math.min(b,Math.max(a,x));

function alertBand(prob:number, threshold:number) {
  if (prob >= Math.max(0.80, threshold)) return {label:'EMERGENCY ALERT', cls:'emergency', icon:<Siren/>};
  if (prob >= 0.50) return {label:'HIGH RISK', cls:'high', icon:<AlertTriangle/>};
  if (prob >= threshold) return {label:'SUSPECT / MONITOR', cls:'suspect', icon:<AlertTriangle/>};
  return {label:'LOW RISK', cls:'low', icon:<CheckCircle2/>};
}

function vitalState(value:any, low:number, high:number) {
  const n=Number(value); if(!Number.isFinite(n)) return 'muted';
  return n<low || n>high ? 'critical' : 'normal';
}

const emptyRow:Row={HR:84,O2Sat:97,Temp:37.1,SBP:122,MAP:82,DBP:64,Resp:17,EtCO2:null,BaseExcess:null,HCO3:null,FiO2:null,pH:null,PaCO2:null,SaO2:null,AST:null,BUN:null,Alkalinephos:null,Calcium:null,Chloride:null,Creatinine:null,Bilirubin_direct:null,Glucose:105,Lactate:1.2,Magnesium:null,Phosphate:null,Potassium:null,Bilirubin_total:null,TroponinI:null,Hct:null,Hgb:null,PTT:null,WBC:8.5,Fibrinogen:null,Platelets:null,Age:65,Gender:0,Unit1:0,Unit2:1,HospAdmTime:-2,ICULOS:1,SepsisLabel:0};

export default function App(){
  const [mode,setMode]=useState<Mode>('dataset');
  const [patientId,setPatientId]=useState('p100013');
  const [patients,setPatients]=useState<{patient_id:string;age:number|null;gender:number|null}[]>([]);
  const [rows,setRows]=useState<Row[]>([]);
  const [prediction,setPrediction]=useState<Prediction|null>(null);
  const [history,setHistory]=useState<Prediction[]>([]);
  const [running,setRunning]=useState(false);
  const [speed,setSpeed]=useState(1500);
  const [error,setError]=useState('');
  const [lastUpdate,setLastUpdate]=useState<Date|null>(null);
  const timer=useRef<number|null>(null);
  const [scenario,setScenario]=useState<'stable'|'deterioration'|'rapid'|'recovery'>('deterioration');

  useEffect(()=>{ api.patients().then(x=>setPatients(x.patients||[])).catch(()=>{}); },[]);
  useEffect(()=>{ if(mode!=='dataset') return; setRunning(false); loadDataset(patientId); },[mode,patientId]);

  async function loadDataset(id:string){
    try{setError(''); const d=await api.patient(id); setRows(d.rows||[]); setPrediction({...d.latest,patient_id:id,hospital_id:'B',timestamp:d.latest.timestamp,probability:d.latest.probability,risk:d.latest.risk,alert:d.latest.alert,threshold:d.threshold,explanation:d.explanation}); setHistory([]); setLastUpdate(new Date());}
    catch(e){setError(String(e));}
  }

  function newPatient(){
    setMode('icu'); setRunning(false); setHistory([]); setPrediction(null); setError('');
    const age=Math.floor(35+Math.random()*55);
    setPatientId(`SIM-${String(Math.floor(100+Math.random()*900))}`);
    setRows([{...emptyRow,Age:age,ICULOS:1}]);
  }

  async function simulateOne(){
    const prev=rows[rows.length-1]||{...emptyRow}; const h=Number(prev.ICULOS||0)+1;
    const t=mode==='icu' ? (h-1) : 0;
    const direction=scenario==='deterioration'?1:scenario==='rapid'?2.2:scenario==='recovery'?-1:0;
    const next={...prev,ICULOS:h,
      HR:clamp(Number(prev.HR??84)+direction*(0.7+0.12*t)+(Math.random()-.5)*5,55,160),
      O2Sat:clamp(Number(prev.O2Sat??97)-direction*(0.10+0.025*t)+(Math.random()-.5)*0.9,75,100),
      Temp:clamp(Number(prev.Temp??37.1)+direction*0.018+(Math.random()-.5)*0.08,35,41),
      SBP:clamp(Number(prev.SBP??122)-direction*(0.35+0.03*t)+(Math.random()-.5)*3,65,180),
      MAP:clamp(Number(prev.MAP??82)-direction*(0.22+0.02*t)+(Math.random()-.5)*2,45,120),
      DBP:clamp(Number(prev.DBP??64)-direction*0.18+(Math.random()-.5)*2,35,110),
      Resp:clamp(Number(prev.Resp??17)+direction*(0.30+0.06*t)+(Math.random()-.5)*1.2,8,45),
      Glucose:clamp(Number(prev.Glucose??105)+direction*0.8+(Math.random()-.5)*6,50,300),
      Lactate:clamp(Number(prev.Lactate??1.2)+direction*(0.035+0.012*t)+(Math.random()-.5)*0.05,0.5,8),
      WBC:clamp(Number(prev.WBC??8.5)+direction*0.06+(Math.random()-.5)*0.25,2,30),
      Age:Number(prev.Age??65),Gender:Number(prev.Gender??0),Unit1:Number(prev.Unit1??0),Unit2:Number(prev.Unit2??1),HospAdmTime:Number(prev.HospAdmTime??-2),SepsisLabel:0
    };
    try{
      const p=await api.predict({patient_id:patientId,hospital_id:mode==='dataset'?'B':'SIM',rows:[...rows.slice(-47),next]});
      const enriched={...next,PredictedRisk:p.probability,SepsisAlert:p.alert?1:0};
      setRows(r=>[...r,enriched].slice(-72)); setPrediction(p); setHistory(x=>[...x,p].slice(-30)); setLastUpdate(new Date()); setError('');
    }catch(e){setError(String(e));setRunning(false);}
  }

  useEffect(()=>{if(!running)return; timer.current=window.setInterval(simulateOne,speed); return()=>{if(timer.current)window.clearInterval(timer.current);};},[running,speed,rows,mode,scenario,patientId]);

  function reset(){setRunning(false);setHistory([]);setPrediction(null);setError(''); if(mode==='icu')setRows([{...emptyRow,ICULOS:1}]); else loadDataset(patientId);}

  const latest=rows[rows.length-1]||emptyRow;
  const chart=useMemo(()=>rows.slice(-40).map((r,i)=>({hour:r.ICULOS??i+1,risk:Number(r.PredictedRisk??0),HR:Number(r.HR??0),O2Sat:Number(r.O2Sat??0),Temp:Number(r.Temp??0),Resp:Number(r.Resp??0)})),[rows]);
  const band=alertBand(prediction?.probability??0,prediction?.threshold??0.14);
  const explain=prediction?.explanation?.features||[];

  return <div className="app">
    <header><div className="brand"><div className="logo"><HeartPulse size={22}/></div><div><b>Sepsis Early-Warning Monitor</b><span>Digital-twin-inspired AI/ML clinical decision-support demonstration</span></div></div><div className="connection"><span className="dot"/> Backend {lastUpdate?'active':'ready'} <small>{lastUpdate?`Updated ${lastUpdate.toLocaleTimeString()}`:'Waiting'}</small></div></header>
    <main>
      <section className="modebar"><button className={mode==='dataset'?'mode active':'mode'} onClick={()=>setMode('dataset')}><Database size={17}/> Dataset Mode</button><button className={mode==='icu'?'mode active':'mode'} onClick={newPatient}><Radio size={17}/> ICU Simulation Mode</button><div className="modecopy">{mode==='dataset'?'Real PhysioNet Hospital B patient data':'New virtual patient — measurements are generated automatically'}</div></section>
      <section className="toolbar">
        {mode==='dataset'?<div className="field"><label>Hospital B patient</label><select value={patientId} onChange={e=>setPatientId(e.target.value)}>{patients.length?patients.slice(0,500).map(p=><option key={p.patient_id}>{p.patient_id}</option>):<option>p100013</option>}</select></div>:<div className="field"><label>Virtual patient</label><div className="readonly">{patientId}</div></div>}
        {mode==='icu'&&<div className="field"><label>Scenario</label><select value={scenario} onChange={e=>setScenario(e.target.value as any)}><option value="stable">Stable</option><option value="deterioration">Gradual deterioration</option><option value="rapid">Rapid deterioration</option><option value="recovery">Recovery</option></select></div>}
        <div className="field"><label>Simulation speed</label><select value={speed} onChange={e=>setSpeed(Number(e.target.value))}><option value={2500}>Slow</option><option value={1500}>Normal</option><option value={700}>Fast</option></select></div>
        <button className="secondary" onClick={reset}><RotateCcw size={16}/> Reset</button><button className={running?'danger':'primary'} onClick={()=>setRunning(v=>!v)}>{running?<><Pause size={16}/> Pause</>:<><Play size={16}/> Start monitoring</>}</button>
      </section>
      {error&&<div className="error"><AlertTriangle size={17}/>{error}</div>}
      <section className="grid top-grid"><div className="card patient-card"><div className="card-title"><span>Patient overview</span><span className="badge">{mode==='dataset'?'Hospital B':'SIMULATED ICU'}</span></div><div className="patient-id">{patientId}</div><div className="demographics"><div><span>Age</span><b>{latest.Age??'—'}</b></div><div><span>Gender</span><b>{latest.Gender===1?'Female':'Male'}</b></div><div><span>ICU hour</span><b>{latest.ICULOS??'—'}</b></div></div></div>
        <div className={`card risk-card ${band.cls}`}><div className="card-title"><span>Current AI prediction</span><Brain size={18}/></div><div className="risk-row"><div className="risk-number">{prediction?`${(prediction.probability*100).toFixed(1)}%`:'—'}</div><div className="risk-label">{prediction?prediction.risk:'NO RESULT'}</div></div><div className="meter"><div style={{width:`${(prediction?.probability||0)*100}%`}}/></div><small>XGBoost · research decision threshold {(prediction?.threshold??0.14).toFixed(2)}</small></div>
        <div className={`card alert-card ${band.cls}`}><div className="card-title"><span>Clinical-style alert</span>{band.icon}</div><div className="big-alert">{prediction?band.label:'WAITING FOR MODEL'}</div><small>{prediction?`Updated ${new Date().toLocaleTimeString()}`:'Start monitoring to generate a prediction'}</small></div></section>
      <section className="grid vitals-grid"><Vital icon={<HeartPulse/>} name="Heart rate" value={latest.HR} unit="bpm" low={60} high={100}/><Vital icon={<Activity/>} name="SpO₂" value={latest.O2Sat} unit="%" low={94} high={100}/><Vital icon={<Thermometer/>} name="Temperature" value={latest.Temp} unit="°C" low={36.1} high={38} digits={1}/><Vital icon={<Wind/>} name="Respiratory rate" value={latest.Resp} unit="/min" low={12} high={20}/><Vital icon={<Activity/>} name="Systolic BP" value={latest.SBP} unit="mmHg" low={90} high={140}/><Vital icon={<Activity/>} name="Lactate" value={latest.Lactate} unit="mmol/L" low={0.5} high={2} digits={1}/></section>
      <section className="grid charts-grid"><ChartCard title="Sepsis-risk trajectory"><ResponsiveContainer width="100%" height="100%"><LineChart data={chart}><XAxis dataKey="hour"/><YAxis domain={[0,1]}/><Tooltip/><Line type="monotone" dataKey="risk" strokeWidth={3} dot={false}/></LineChart></ResponsiveContainer></ChartCard><ChartCard title="Live vital trends"><ResponsiveContainer width="100%" height="100%"><LineChart data={chart}><XAxis dataKey="hour"/><YAxis/><Tooltip/><Line type="monotone" dataKey="HR" strokeWidth={2} dot={false}/><Line type="monotone" dataKey="O2Sat" strokeWidth={2} dot={false}/><Line type="monotone" dataKey="Resp" strokeWidth={2} dot={false}/></LineChart></ResponsiveContainer></ChartCard></section>
      <section className="grid bottom-grid"><div className="card explain-card"><div className="card-title"><span>Why did the model make this prediction?</span><span className="badge purple">SHAP</span></div>{explain.length?<div className="features">{explain.slice(0,8).map((f,i)=><div className="feature" key={i}><div><b>{f.feature}</b><span>value: {f.value==null?'missing':Number(f.value).toFixed(3)}</span></div><div className={f.shap>=0?'contrib up':'contrib down'}>{f.shap>=0?'+':''}{f.shap.toFixed(3)}<small>{f.direction}</small></div></div>)}</div>:<div className="empty">Run the model to see actual SHAP feature attributions returned by the backend.</div>}</div>
        <div className="card timeline-card"><div className="card-title"><span>Prediction & alert timeline</span><span className="muted">Latest events</span></div><div className="timeline">{history.slice().reverse().slice(0,10).map((p,i)=><div className="event" key={i}><span className={`event-dot ${alertBand(p.probability,p.threshold).cls}`}/><div><b>ICU hour {p.timestamp}</b><span>{(p.probability*100).toFixed(1)}% · {alertBand(p.probability,p.threshold).label}</span></div></div>)}</div></div></section>
      <div className="disclaimer"><ShieldCheck size={17}/><span><b>Demonstration only.</b> This prototype provides model-based risk estimates and alerts. It is not an autonomous diagnosis system; clinical decisions require qualified professional judgment. Alert bands are demonstration labels, not clinical thresholds.</span></div>
    </main>
  </div>
}

function ChartCard({title,children}:{title:string;children:any}){return <div className="card chart-card"><div className="card-title"><span>{title}</span><span className="muted">Recent ICU hours</span></div><div className="chart">{children}</div></div>}
function Vital({icon,name,value,unit,low,high,digits=0}:{icon:any,name:string,value:any,unit:string,low:number,high:number,digits?:number}){const n=Number(value);const state=vitalState(value,low,high);return <div className={`card vital ${state}`}><div className="vital-head">{icon}<span>{name}</span></div><div className="vital-value">{Number.isFinite(n)?n.toFixed(digits):'—'}<small>{unit}</small></div><span className="range">Reference {low}–{high}</span></div>}
