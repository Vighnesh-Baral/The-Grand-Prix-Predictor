import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, ArrowRight, BarChart3, CalendarDays, CarFront, CheckCircle2,
  ChevronDown, ChevronRight, CircleHelp, CircuitBoard, Clock3, Gauge,
  Flag, LayoutDashboard, Menu, RefreshCw, Shield, Timer, Trophy, Users,
  X, Zap, AlertTriangle, Database, Target, Settings2
} from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import './styles.css';

const API = import.meta.env.VITE_API_BASE || '';
const NAV = [
  { group: 'RACE WEEKEND', items: [['Race Centre', LayoutDashboard], ['Pole Predictor', Target], ['Race Predictor', Flag], ['Practice Intelligence', Timer]] },
  { group: 'PERFORMANCE', items: [['Constructor Intelligence', CarFront], ['Driver Intelligence', Users], ['Circuit Intelligence', CircuitBoard]] },
  { group: 'RESEARCH', items: [['Season Standings', Trophy], ['Model Lab', BarChart3]] },
];

const PHOTO = {
  race: 'https://thumb.wikimedia.org/wikipedia/commons/thumb/0/0f/FIA_F1_Austria_2023_Nr._63_%281%29.jpg/1280px-FIA_F1_Austria_2023_Nr._63_%281%29.jpg',
  circuit: 'https://thumb.wikimedia.org/wikipedia/commons/thumb/5/59/Grande_Pr%C3%AAmio_S%C3%A3o_Paulo_de_F%C3%B3rmula_1_%2852497080967%29.jpg/1280px-Grande_Pr%C3%AAmio_S%C3%A3o_Paulo_de_F%C3%B3rmula_1_%2852497080967%29.jpg',
  pit: 'https://thumb.wikimedia.org/wikipedia/commons/thumb/5/50/Aston_Martin_pit_USGP_2021.jpg/960px-Aston_Martin_pit_USGP_2021.jpg',
  duel: 'https://thumb.wikimedia.org/wikipedia/commons/thumb/8/84/2012_Italian_GP_-_Alonso-Vettel.jpg/1280px-2012_Italian_GP_-_Alonso-Vettel.jpg',
  pack: 'https://thumb.wikimedia.org/wikipedia/commons/thumb/2/2f/Formula_one.jpg/1280px-Formula_one.jpg',
  filmCar: 'https://thumb.wikimedia.org/wikipedia/commons/thumb/a/af/APEX_MOVIE_F1CAR.jpg/1280px-APEX_MOVIE_F1CAR.jpg',
};
const PHOTO_CREDITS = {
  race: 'Lukas Raich · CC BY-SA 4.0',
  circuit: 'Governo do Estado de São Paulo / Beto Issa · CC BY 2.0',
  pit: 'Declan M Martin · Public domain',
  duel: 'Francesco Crippa · CC BY 2.0',
  pack: 'Rick Dikeman · CC BY-SA 3.0',
  filmCar: 'Mac Edmunds · CC BY-SA 4.0',
};
const PHOTO_SOURCES = {
  race: 'https://commons.wikimedia.org/wiki/File:FIA_F1_Austria_2023_Nr._63_(1).jpg',
  circuit: 'https://commons.wikimedia.org/wiki/File:Grande_Pr%C3%AAmio_S%C3%A3o_Paulo_de_F%C3%B3rmula_1_(52497080967).jpg',
  pit: 'https://commons.wikimedia.org/wiki/File:Aston_Martin_pit_USGP_2021.jpg',
  duel: 'https://commons.wikimedia.org/wiki/File:2012_Italian_GP_-_Alonso-Vettel.jpg',
  pack: 'https://commons.wikimedia.org/wiki/File:Formula_one.jpg',
  filmCar: 'https://commons.wikimedia.org/wiki/File:APEX_MOVIE_F1CAR.jpg',
};
const COUNTRY_CODES = {
  'Australia':'au','Austria':'at','Azerbaijan':'az','Bahrain':'bh','Belgium':'be','Brazil':'br','Canada':'ca','China':'cn',
  'France':'fr','Germany':'de','Hungary':'hu','India':'in','Italy':'it','Japan':'jp','Malaysia':'my','Mexico':'mx','Monaco':'mc',
  'Netherlands':'nl','Portugal':'pt','Qatar':'qa','Russia':'ru','Saudi Arabia':'sa','Singapore':'sg','South Africa':'za',
  'Spain':'es','Switzerland':'ch','Turkey':'tr','United Arab Emirates':'ae','United Kingdom':'gb','United States':'us',
  'United States of America':'us','United States':'us','USA':'us','US':'us','Great Britain':'gb','Britain':'gb','UK':'gb','U.K.':'gb','U.S.A.':'us','UAE':'ae','Korea':'kr','South Korea':'kr'
};
const DRIVER_FLAGS = {
  max_verstappen:'nl', perez:'mx', hamilton:'gb', russell:'gb', leclerc:'mc', sainz:'es', norris:'gb', piastri:'au', alonso:'es', stroll:'ca',
  gasly:'fr', ocon:'fr', albon:'th', sargeant:'us', tsunoda:'jp', ricciardo:'au', bottas:'fi', zhou:'cn', hulkenberg:'de', magnussen:'dk',
  schumacher:'de', latifi:'ca', vettel:'de', raikkonen:'fi', giovinazzi:'it', kubica:'pl', grosjean:'fr', antonelli:'it', bearman:'gb',
  colapinto:'ar', doohan:'au', hadjar:'fr', bortoleto:'br', lawson:'nz', de_vries:'nl', vandoorne:'be', hartley:'nz', sutil:'de',
  di_resta:'gb', buemi:'ch', kovalainen:'fi', petrov:'ru', liuzzi:'it', alguersuari:'es', barrichello:'br', button:'gb', webber:'au'
};
const CONSTRUCTOR_FLAGS = {
  red_bull:'at', mercedes:'de', ferrari:'it', mclaren:'gb', alpine:'fr', aston_martin:'gb', williams:'gb', haas:'us', rb:'it',
  alphatauri:'it', toro_rosso:'it', sauber:'ch', alfa:'it', alfa_romeo:'ch', force_india:'in', racing_point:'gb', renault:'fr',
  lotus:'gb', caterham:'my', marussia:'ru', manor:'gb', virgin:'gb', brawn:'gb', toyota:'jp', honda:'jp', bmw_sauber:'de',
  super_aguri:'jp', spyker:'nl', jordan:'ie', minardi:'it', jaguar:'gb', bar:'gb', prost:'fr', arrows:'gb'
};
const DRIVER_NAME_FLAGS = {
  'max verstappen':'nl','sergio perez':'mx','lewis hamilton':'gb','george russell':'gb','charles leclerc':'mc','carlos sainz':'es',
  'lando norris':'gb','oscar piastri':'au','fernando alonso':'es','lance stroll':'ca','pierre gasly':'fr','esteban ocon':'fr',
  'alexander albon':'th','logan sargeant':'us','yuki tsunoda':'jp','daniel ricciardo':'au','valtteri bottas':'fi','zhou guanyu':'cn',
  'nico hulkenberg':'de','kevin magnussen':'dk','kimi antonelli':'it','oliver bearman':'gb','franco colapinto':'ar','liam lawson':'nz',
  'gabriel bortoleto':'br','isack hadjar':'fr','sebastian vettel':'de','kimi raikkonen':'fi','jenson button':'gb','mark webber':'au'
};
const CONSTRUCTOR_NAME_FLAGS = {
  'red bull':'at','red bull racing':'at','mercedes':'de','ferrari':'it','mclaren':'gb','alpine':'fr','aston martin':'gb',
  'williams':'gb','haas':'us','rb':'it','visa cash app rb':'it','alphatauri':'it','toro rosso':'it','sauber':'ch',
  'alfa romeo':'ch','force india':'in','racing point':'gb','renault':'fr','lotus':'gb','caterham':'my','brawn':'gb'
};
const driverFlagFor = row => DRIVER_FLAGS[String(row?.driver_id || '').toLowerCase()] || DRIVER_NAME_FLAGS[String(row?.driver_name || row?.driver || '').trim().toLowerCase()];
const constructorFlagFor = row => CONSTRUCTOR_FLAGS[String(row?.constructor_id || '').toLowerCase()] || CONSTRUCTOR_NAME_FLAGS[String(row?.constructor_name || row?.constructor || '').trim().toLowerCase()];
const countryCode = value => COUNTRY_CODES[String(value || '').trim()] || null;
function CountryFlag({ country, code, className = '' }) {
  const iso = code || countryCode(country);
  return iso ? <img className={`country-flag ${className}`} src={`https://flagcdn.com/w40/${iso}.png`} srcSet={`https://flagcdn.com/w80/${iso}.png 2x`} alt={`${country || iso.toUpperCase()} flag`} loading="lazy" decoding="async"/> : null;
}
function PhotoCredit({ photo, label }) {
  return <span className="photo-credit"><span>{label || PHOTO_CREDITS[photo]}</span><a href={PHOTO_SOURCES[photo]} target="_blank" rel="noreferrer">Source & licence</a></span>;
}
const HEADER_ART = [
  [/qualifying/i, 'race', 'A Formula One car attacking the kerbs during qualifying'],
  [/grand prix|race weekend/i, 'pack', 'Formula One cars negotiating a circuit'],
  [/practice/i, 'pit', 'A Formula One team working on a car in the pit lane'],
  [/car performance|constructor/i, 'filmCar', 'A Formula One car in motion at Silverstone'],
  [/driver/i, 'duel', 'Two Formula One cars on the pit straight'],
  [/circuit/i, 'circuit', 'Aerial view of the Autódromo José Carlos Pace at Interlagos'],
  [/championship/i, 'duel', 'Formula One cars during a Grand Prix weekend'],
  [/model validation|model/i, 'race', 'Formula One car on track'],
];

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...(options.headers || {}) },
  });
  const data = await response.json().catch(() => ({ detail: 'The server returned an unreadable response.' }));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}
const fmt = (value, digits = 1) => value === null || value === undefined || value === '' || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(digits);
const pct = value => `${(Number(value || 0) * 100).toFixed(1)}%`;
const dateFmt = value => value ? new Date(value).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' }) : 'Date unavailable';
const titleEvent = event => event ? `${event.race_name} · ${event.season}` : 'No upcoming Grand Prix found';

function Brand({ onClick }) {
  return <button className="brand" onClick={onClick} aria-label="The Grand Prix Predictor home"><span className="brand-mark gpp-brand-mark"><span>GPP</span><i/></span><span className="brand-copy"><b>THE GRAND PRIX <i>PREDICTOR</i></b></span></button>;
}
function PageHeader({ kicker, title, detail, right }) {
  const art = HEADER_ART.find(([pattern]) => pattern.test(kicker || '')) || HEADER_ART[0];
  const [, photo, alt] = art;
  return <>
    <div className="page-header"><div className="page-header-copy"><div className="eyebrow"><span/>{kicker}</div><h1>{title}</h1><p>{detail}</p></div>{right && <div className="page-header-right">{right}</div>}</div>
    <figure className="page-header-visual"><img src={PHOTO[photo]} alt={alt} loading="lazy" decoding="async"/><PhotoCredit photo={photo}/></figure>
  </>;
}

function Metric({ label, value, note, icon: Icon }) {
  return <div className="metric-card"><div className="metric-top"><span>{label}</span><Icon size={17}/></div><strong>{value ?? '—'}</strong><small>{note}</small></div>;
}
function StatusPill({ children, tone = 'neutral' }) { return <span className={`status-pill ${tone}`}>{children}</span>; }
function Empty({ title, body }) { return <div className="empty-state"><CircleHelp size={24}/><b>{title}</b><p>{body}</p></div>; }

function LandingPage({ onEnter }) {
  const [pointer, setPointer] = useState({ x: 0, y: 0 });
  const [aeroMode, setAeroMode] = useState('flow');
  const carRef = useRef(null);
  const handlePointer = event => {
    const bounds = event.currentTarget.getBoundingClientRect();
    setPointer({
      x: ((event.clientX - bounds.left) / bounds.width - 0.5) * 2,
      y: ((event.clientY - bounds.top) / bounds.height - 0.5) * 2,
    });
  };
  useEffect(() => {
    const viewer = carRef.current;
    if (!viewer) return undefined;
    const applyMaterials = () => {
      const materials = viewer.model?.materials || [];
      materials.forEach(material => {
        const pbr = material.pbrMetallicRoughness;
        if (!pbr) return;
        // The supplied GLB ships without a base-colour texture/factor. Set an
        // intentional graphite finish instead of allowing model-viewer to show white.
        pbr.setBaseColorFactor([0.13, 0.15, 0.18, 1]);
        pbr.setMetallicFactor(0.72);
        pbr.setRoughnessFactor(0.31);
      });
    };
    const motionPreference = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    const syncMotionPreference = () => {
      if (motionPreference?.matches) viewer.removeAttribute('auto-rotate');
      else viewer.setAttribute('auto-rotate', '');
    };
    viewer.addEventListener('load', applyMaterials);
    motionPreference?.addEventListener?.('change', syncMotionPreference);
    syncMotionPreference();
    if (viewer.loaded) applyMaterials();
    return () => {
      viewer.removeEventListener('load', applyMaterials);
      motionPreference?.removeEventListener?.('change', syncMotionPreference);
    };
  }, []);

  return <main className="gpp-landing">
    <div className="landing-grid" aria-hidden="true" />
    <div className="landing-redline" aria-hidden="true" />

    <section className="landing-hero">
      <div className="landing-copy">
        <div className="landing-kicker"><span className="kicker-line"/> INDEPENDENT MOTORSPORT INTELLIGENCE</div>
        <h1>THE GRAND<br/><span>PRIX</span><em>PREDICTOR</em></h1>
        <p className="landing-description">A race is never decided by one number.<br/>The Grand Prix Predictor brings pace, form, circuit context and probability into one place.</p>
        <div className="landing-actions">
          <button className="landing-enter" onClick={onEnter}><span>EXPLORE THE PLATFORM</span><ArrowRight size={17}/></button>
          <a className="landing-action-note" href="#race-method">HOW THE ANALYSIS WORKS</a>
        </div>
        <div className="landing-hero-caption">FORMULA ONE · PERFORMANCE ANALYSIS · PREDICTIVE MODELS</div>
      </div>

      <div className="landing-car-stage" onPointerMove={handlePointer} onPointerLeave={() => setPointer({ x: 0, y: 0 })} style={{ '--pointer-x': `${pointer.x * 5}px`, '--pointer-y': `${pointer.y * 4}px` }}>
        <div className="car-aura" />
        <div className="car-orbit orbit-one" />
        <div className="car-orbit orbit-two" />
        <div className="car-model-float">
          <model-viewer ref={carRef} class="gpp-car-model" src="/models/f1-car.glb" alt="Animated 3D concept Formula racing car in graphite finish"
            auto-rotate auto-rotate-delay="0" rotation-per-second="8deg" camera-controls disable-zoom disable-pan
            interaction-prompt="none" camera-orbit="-28deg 68deg 120%" shadow-intensity="1.35" shadow-softness="0.9"
            exposure="0.86" environment-image="neutral" tone-mapping="aces" ar="false" loading="eager" />
        </div>
        <div className={`aero-overlay ${aeroMode}`} aria-hidden="true">
          {aeroMode === 'flow' ? <svg viewBox="0 0 760 520" preserveAspectRatio="none">
            <defs><linearGradient id="flowStroke" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stopColor="#36b9ee" stopOpacity="0"/><stop offset=".45" stopColor="#58d7ff" stopOpacity=".95"/><stop offset="1" stopColor="#ed2638" stopOpacity=".68"/></linearGradient></defs>
            {[0,1,2,3,4,5,6,7,8].map(i => <path key={i} className="flow-line" style={{ '--flow-delay': `${i * -0.72}s` }} d={`M -20 ${88+i*38} C 115 ${45+i*36}, 170 ${150+i*21}, 310 ${210+i*11} S 515 ${325+i*5}, 780 ${125+i*30}`} />)}
            {[0,1,2,3,4,5].map(i => <circle key={i} r="3.3" fill="#75dcff"><animateMotion dur={`${5.5+i*.45}s`} repeatCount="indefinite" begin={`${-i*.9}s`} path={`M -20 ${88+i*48} C 115 ${45+i*36}, 170 ${150+i*21}, 310 ${210+i*11} S 515 ${325+i*5}, 780 ${125+i*30}`}/></circle>)}
          </svg> : <div className="pressure-field"><i/><i/><i/><i/></div>}
        </div>
        <div className="aero-switch" role="group" aria-label="Aerodynamic visualisation mode">
          <button className={aeroMode === 'flow' ? 'active' : ''} onClick={() => setAeroMode('flow')}>AIRFLOW</button>
          <button className={aeroMode === 'pressure' ? 'active' : ''} onClick={() => setAeroMode('pressure')}>PRESSURE FIELD</button>
        </div>
      </div>
    </section>

    <div className="landing-rule"><span>01</span><i/> THE DETAILS BEHIND THE RESULT</div>

    <section className="editorial-intro" id="race-method">
      <div className="editorial-heading"><span className="section-number">01 / THE RACE</span><h2>Speed is visible.<br/><em>The reasons are not.</em></h2></div>
      <div className="editorial-copy"><p>Lap time is the outcome of hundreds of small decisions: how a car loads its tyres, how a driver manages a stint, how much straight-line speed a circuit rewards, and how cleanly a team executes.</p><p>GPP is built to examine those signals together. It uses historical race records and available weekend data to form a structured view of performance—not a promise of what will happen on Sunday.</p></div>
      <figure className="editorial-photo"><img src={PHOTO.race} alt="George Russell's Mercedes Formula One car at the 2023 Austrian Grand Prix" loading="lazy"/><PhotoCredit photo="race"/></figure>
    </section>

    <section className="physics-section">
      <div className="section-intro"><span className="section-number">02 / WHAT SHAPES A LAP</span><h2>Three forces.<br/><em>One competitive picture.</em></h2><p>Performance is a relationship between the car, the circuit and the way the race unfolds.</p></div>
      <div className="physics-grid">
        <article className="physics-card physics-card-aero"><img src={PHOTO.filmCar} alt="Formula One car on track at Silverstone" loading="lazy"/><div className="physics-card-content"><span>01 · AERODYNAMICS</span><h3>Air is part of the car.</h3><p>Downforce, drag and airflow determine how confidently a car can turn and how much speed it carries down a straight.</p></div><PhotoCredit photo="filmCar"/></article>
        <article className="physics-card physics-card-tyre"><img src={PHOTO.pit} alt="Aston Martin pit crew working on the car during the 2021 United States Grand Prix" loading="lazy"/><div className="physics-card-content"><span>02 · TYRE MANAGEMENT</span><h3>Grip changes over a stint.</h3><p>Compound, tyre age, temperature and driving style affect the pace a driver can sustain across a run.</p></div><PhotoCredit photo="pit"/></article>
        <article className="physics-card physics-card-circuit"><img src={PHOTO.circuit} alt="Aerial view of Interlagos, the Autódromo José Carlos Pace" loading="lazy"/><div className="physics-card-content"><span>03 · CIRCUIT CONTEXT</span><h3>Every track asks a different question.</h3><p>Layout, corner sequences and overtaking opportunities change the value of pace and track position.</p></div><PhotoCredit photo="circuit"/></article>
      </div>
    </section>

    <section className="aero-editorial">
      <div className="aero-editorial-image">
        <img src={PHOTO.race} alt="Formula One car generating aerodynamic load on track" loading="lazy"/>
        <svg className="aero-editorial-lines" viewBox="0 0 900 460" preserveAspectRatio="none" aria-hidden="true">
          {Array.from({length:9},(_,i)=><path key={i} className="flow-line" style={{'--flow-delay':`${i*-.8}s`}} d={`M -20 ${70+i*35} C 140 ${30+i*27}, 300 ${130+i*26}, 490 ${200+i*8} S 690 ${280+i*12}, 920 ${95+i*30}`}/>)}
        </svg>
        <div className="aero-image-caption">ILLUSTRATIVE AIRFLOW PATHS · NOT A CFD SOLUTION</div>
        <PhotoCredit photo="race"/>
      </div>
      <div className="aero-editorial-copy"><span className="section-number">03 / AERODYNAMIC VISUALISATION</span><h2>Read the air<br/><em>around the car.</em></h2><p>The interactive car view uses animated streamlines and a pressure-style colour field to make aerodynamic ideas easier to explore. These overlays are visual explanations, not solver-generated measurements.</p><button className="text-link" onClick={() => onEnter('Constructor Intelligence')}>Explore the analysis workspace <ArrowRight size={16}/></button></div>
    </section>

    <section className="weekend-section">
      <div className="section-intro"><span className="section-number">04 / A GRAND PRIX WEEKEND</span><h2>Form develops<br/><em>session by session.</em></h2><p>The evidence available before qualifying is different from the evidence available before the race. GPP keeps those stages separate.</p></div>
      <div className="weekend-timeline">
        {[
          ['FP1','Baseline','Initial balance, tyre behaviour and first representative laps.'],
          ['FP2','Longer runs','Race-pace evidence, stint patterns and setup direction.'],
          ['FP3','Final practice','The final practice picture before qualifying begins.'],
          ['QUALIFYING','One lap','A single-lap contest shaped by preparation and track evolution.'],
          ['RACE','Race day','Grid position, pace, strategy, reliability and uncertainty.'],
        ].map(([code,name,copy],i)=><article className="weekend-stage" key={code}><span className="stage-index">0{i+1}</span><div className="stage-line"/><div><small>{code}</small><h3>{name}</h3><p>{copy}</p></div></article>)}
      </div>
    </section>

    <section className="model-section">
      <div className="model-section-copy"><span className="section-number">05 / THE MODEL</span><h2>Probability,<br/><em>with context.</em></h2><p>The prediction system draws on historical outcomes, current-season performance and the information available at the selected stage of a race weekend. Its job is to rank possibilities and show uncertainty—not to manufacture certainty.</p><p>Every forecast should be read alongside its data coverage, assumptions and model limitations.</p><button className="landing-enter" onClick={() => onEnter('Pole Predictor')}><span>OPEN THE PREDICTORS</span><ArrowRight size={17}/></button></div>
      <figure className="model-section-photo"><img src={PHOTO.duel} alt="Fernando Alonso and Sebastian Vettel at the 2012 Italian Grand Prix" loading="lazy"/><PhotoCredit photo="duel"/></figure>
    </section>

    <section className="platform-preview">
      <div className="section-intro"><span className="section-number">06 / THE WORKSPACE</span><h2>One weekend.<br/><em>Several ways to read it.</em></h2><p>Move between predictions, driver and constructor analysis, circuit history and model validation.</p></div>
      <div className="preview-grid">
        <button className="preview-card preview-card-wide" onClick={() => onEnter('Race Centre')}><img src={PHOTO.pack} alt="Formula One cars racing through a circuit" loading="lazy"/><span>01 / RACE CENTRE</span><b>Set the weekend in context</b><ArrowRight size={18}/></button>
        <button className="preview-card" onClick={() => onEnter('Race Predictor')}><img src={PHOTO.race} alt="Formula One car attacking the track" loading="lazy"/><span>02 / PREDICTION</span><b>Qualifying and race outcomes</b><ArrowRight size={18}/></button>
        <button className="preview-card" onClick={() => onEnter('Driver Intelligence')}><img src={PHOTO.pit} alt="A team preparing a Formula One car" loading="lazy"/><span>03 / PERFORMANCE</span><b>Driver, constructor and practice</b><ArrowRight size={18}/></button>
        <button className="preview-card preview-card-wide" onClick={() => onEnter('Circuit Intelligence')}><img src={PHOTO.circuit} alt="Interlagos circuit from above" loading="lazy"/><span>04 / CIRCUITS & RESEARCH</span><b>Track history and model evidence</b><ArrowRight size={18}/></button>
      </div>
    </section>

    <section className="landing-final-cta">
      <img src={PHOTO.pack} alt="Formula One race pack" loading="lazy"/>
      <div><span className="section-number">THE GRAND PRIX PREDICTOR</span><h2>Look beyond<br/>the finishing order.</h2><button className="landing-enter" onClick={onEnter}><span>ENTER THE PLATFORM</span><ArrowRight size={17}/></button></div>
      <PhotoCredit photo="pack"/>
    </section>

    <footer className="landing-footer"><span>THE GRAND PRIX PREDICTOR</span><span>Independent motorsport analysis · Unofficial and not affiliated with Formula 1 or the FIA</span><a href="/credits" onClick={e=>{e.preventDefault();document.getElementById('credits-note')?.scrollIntoView({behavior:'smooth'});}}>IMAGE CREDITS</a></footer>
    <div id="credits-note" className="landing-credits-note">Photographic credits and licence links are documented in the project’s CREDITS.md file.</div>
  </main>;
}

function App() {
  const [showPlatform, setShowPlatform] = useState(false);
  const [active, setActive] = useState('Race Centre');
  const [overview, setOverview] = useState(null);
  const [events, setEvents] = useState([]);
  const [archiveEvents, setArchiveEvents] = useState([]);
  const [circuits, setCircuits] = useState([]);
  const [selected, setSelected] = useState(null);
  const [constructorSelected, setConstructorSelected] = useState(null);
  const [practiceSelected, setPracticeSelected] = useState(null);
  const [practiceData, setPracticeData] = useState(null);
  const [poleForecastStage, setPoleForecastStage] = useState('after_free_practice');
  const [poleCutoff, setPoleCutoff] = useState('FP3');
  const [raceStage, setRaceStage] = useState('before_qualifying');
  const [eventDetail, setEventDetail] = useState(null);
  const [standings, setStandings] = useState(null);
  const [history, setHistory] = useState(null);
  const [constructorIntel, setConstructorIntel] = useState(null);
  const [constructorIntelError, setConstructorIntelError] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictionError, setPredictionError] = useState('');
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);
  const [season, setSeason] = useState(null);
  const [dataRevision, setDataRevision] = useState(0);

  const refresh = useCallback(async () => {
    setLoading(true); setError('');
    try {
      // CSV ingestion runs outside the API process. Clear its one-entry cache so
      // the refresh button actually sees newly collected FP/lap/stint files.
      await request('/api/dev/reload-data', { method: 'POST' }).catch(() => null);
      const [o, e, archive, circuitData] = await Promise.all([
        request('/api/overview'),
        request('/api/events?upcoming_only=true'),
        request('/api/events?upcoming_only=false'),
        request('/api/circuits?first_season=2018'),
      ]);
      setOverview(o); setEvents(e.events || []); setArchiveEvents(archive.events || []);
      setCircuits(circuitData.circuits || []);
      const defaultEvent = o.upcoming_event || (e.events || [])[0] || null;
      const defaultArchiveEvent = defaultEvent || (archive.events || [])[0] || null;
      const latestCompletedEvent = (archive.events || []).find(item => item.has_race_result) || defaultArchiveEvent;
      setSelected(current => current || defaultEvent);
      setConstructorSelected(current => current || defaultArchiveEvent);
      setPracticeSelected(current => current || latestCompletedEvent);
      setSeason(o.current_season || null);
      setDataRevision(value => value + 1);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!selected) { setEventDetail(null); return; }
    setEventDetail(null);
    request(`/api/events/${selected.season}/${selected.round}`).then(data => {
      setEventDetail(data);
      const available = new Set((data.practice_sessions || []).filter(row => row.available).map(row => row.session));
      const requiredByCutoff = { FP1: ['FP1'], FP2: ['FP1', 'FP2'], FP3: ['FP1', 'FP2', 'FP3'] };
      const latest = ['FP3', 'FP2', 'FP1'].find(cutoff => requiredByCutoff[cutoff].every(code => available.has(code)));
      if (latest) setPoleCutoff(latest);
    }).catch(err => setError(err.message));
  }, [selected, dataRevision]);
  useEffect(() => {
    if (!season) return;
    request(`/api/standings?season=${season}`).then(setStandings).catch(() => setStandings(null));
  }, [season]);
  useEffect(() => { request('/api/model/metrics').then(setMetrics).catch(() => setMetrics(null)); }, []);
  useEffect(() => { setPrediction(null); setPredictionError(''); }, [active, selected]);
  useEffect(() => {
    if (active !== 'Constructor Intelligence' || !constructorSelected) return;
    setConstructorIntel(null); setConstructorIntelError('');
    request(`/api/intelligence/constructor-pace?season=${constructorSelected.season}&round=${constructorSelected.round}&as_of_session=FP3`)
      .then(setConstructorIntel).catch(err => setConstructorIntelError(err.message));
  }, [active, constructorSelected, dataRevision]);
  useEffect(() => {
    if (active !== 'Practice Intelligence' || !practiceSelected) return;
    setPracticeData(null);
    request(`/api/practice?season=${practiceSelected.season}&round=${practiceSelected.round}`)
      .then(setPracticeData).catch(err => setPracticeData({ available: false, rows: [], sessions: [], detail: err.message }));
  }, [active, practiceSelected, dataRevision]);
  useEffect(() => {
    if (selected) setRaceStage(selected.has_qualifying_result ? 'after_qualifying' : 'before_qualifying');
  }, [selected]);

  const chooseEvent = (event) => { setSelected(event); setMenuOpen(false); };
  const chooseConstructorEvent = (event) => { setConstructorSelected(event); setMenuOpen(false); };
  const runPrediction = async (kind) => {
    if (!selected) return;
    setLoadingPrediction(true); setPrediction(null); setPredictionError('');
    try {
      const projectedGridCutoff = poleForecastStage === 'before_free_practice' ? 'PRE_FP' : poleCutoff;
      const requestBody = {
        season: selected.season,
        round: selected.round,
        as_of_session: kind === 'pole' ? projectedGridCutoff : (raceStage === 'before_qualifying' ? projectedGridCutoff : 'FP3'),
      };
      if (kind === 'race') requestBody.race_stage = raceStage;
      const data = await request(`/api/predictions/${kind}`, { method: 'POST', body: JSON.stringify(requestBody) });
      setPrediction(data);
    } catch (err) { setPredictionError(err.message); }
    finally { setLoadingPrediction(false); }
  };

  const title = active;
  const currentSeasonEvents = useMemo(() => events.filter(e => !season || e.season === season), [events, season]);
  const localSeason = season || overview?.current_season;
  const next = overview?.upcoming_event;
  const practiceRows = practiceData?.rows || [];
  const raceResults = eventDetail?.race_results || [];
  const qualiRows = eventDetail?.qualifying || [];

  if (!showPlatform) return <LandingPage onEnter={(page = 'Race Centre') => { setActive(page); setShowPlatform(true); }} />;

  return <div className="app-shell">
    <header className="workspace-nav">
      <div className="workspace-brand"><Brand onClick={() => { setShowPlatform(false); setActive('Race Centre'); }}/></div>
      <nav className="workspace-nav-links" aria-label="Platform sections">
        {NAV.flatMap(group => group.items).map(([label, Icon]) => <button key={label} className={`workspace-nav-link ${active === label ? 'active' : ''}`} onClick={() => setActive(label)}><Icon size={15}/><span>{label}</span></button>)}
      </nav>
      <button className="workspace-home" onClick={() => { setShowPlatform(false); setActive('Race Centre'); }} title="Back to introduction">INTRO <ArrowRight size={14}/></button>
    </header>
    <main className="main-area">
      <div className="content-wrap">
        {error && <div className="error-banner"><AlertTriangle size={17}/><span>{error}</span><button onClick={() => setError('')}><X size={15}/></button></div>}
        {loading && !overview ? <div className="loading-state"><div className="loader"/><b>Loading race intelligence</b><span>Reading local F1 tables and model status.</span></div> : <>
          {active === 'Race Centre' && <>
            <PageHeader kicker="RACE WEEKEND / RACE CENTRE" title="The weekend, in context." detail="A single view of the next Grand Prix, the available historical record and the analysis tools for the sessions ahead."/>
            <section className="race-feature">
              <div className="race-feature-copy">
                <div className="race-feature-overline">NEXT ON THE CALENDAR</div>
                <div className="race-feature-titleline"><CountryFlag country={next?.country}/><span>{next?.country || 'Formula One calendar'}</span>{next?.season && <small>{next.season} · ROUND {String(next.round || '—').padStart(2,'0')}</small>}</div>
                <h2>{next?.race_name || 'No upcoming Grand Prix found'}</h2>
                <p>{next ? `${next.circuit_name || 'Circuit details unavailable'} · ${next.locality || ''}${next.country ? `, ${next.country}` : ''}` : 'The local calendar does not currently contain a future race with a verified start time.'}</p>
                <div className="race-feature-facts">
                  <div><small>RACE DATE</small><b>{dateFmt(next?.start_utc)}</b></div>
                  <div><small>ROUND</small><b>{next?.round ? String(next.round).padStart(2,'0') : '—'}</b></div>
                  <div><small>WEEKEND DATA</small><b>{next?.has_qualifying_result ? 'Qualifying on record' : 'Awaiting session records'}</b></div>
                </div>
                <div className="race-feature-actions"><button className="button-dark" onClick={() => {chooseEvent(next);setActive('Pole Predictor')}} disabled={!next}>Open pole predictor <ArrowRight size={16}/></button><button className="button-light" onClick={() => {chooseEvent(next);setActive('Practice Intelligence')}} disabled={!next}>Practice archive <Timer size={15}/></button></div>
              </div>
              <figure className="race-feature-image">
                <img src={/interlagos|são paulo|brazil/i.test(`${next?.country || ''} ${next?.race_name || ''} ${next?.circuit_name || ''}`) ? PHOTO.circuit : PHOTO.race} alt={/interlagos|são paulo|brazil/i.test(`${next?.country || ''} ${next?.race_name || ''} ${next?.circuit_name || ''}`) ? 'Aerial view of Interlagos' : 'Formula One car on track'}/>
                <div className="race-feature-image-label"><b>{next?.circuit_name || 'FORMULA ONE'}</b><span>{next?.locality || next?.country || 'RACE WEEKEND'}</span></div>
                <PhotoCredit photo={/interlagos|são paulo|brazil/i.test(`${next?.country || ''} ${next?.race_name || ''} ${next?.circuit_name || ''}`) ? 'circuit' : 'race'}/>
              </figure>
            </section>

            <section className="race-data-strip" aria-label="Local archive coverage">
              <div><strong>{overview?.race_events?.toLocaleString() ?? '—'}</strong><span>Grand Prix calendar entries</span></div>
              <div><strong>{overview?.race_result_rows?.toLocaleString() ?? '—'}</strong><span>Race classifications</span></div>
              <div><strong>{overview?.qualifying_rows?.toLocaleString() ?? '—'}</strong><span>Qualifying records</span></div>
              <div><strong>{overview?.practice_rows?.toLocaleString() ?? '—'}</strong><span>Practice records</span></div>
            </section>

            <section className="dashboard-widget-grid">
              <article className="dashboard-widget widget-weekend">
                <div className="widget-heading"><span>THE WEEKEND</span><b>Five stages, one evolving picture</b></div>
                <div className="session-sequence">{[['01','FP1'],['02','FP2'],['03','FP3'],['04','QUALIFYING'],['05','RACE']].map(([n,label],i)=><div className="session-node" key={label}><span>{n}</span><b>{label}</b>{i<4&&<i/>}</div>)}</div>
                <p>Session cut-offs matter. A forecast made before practice should not quietly use information that only exists after qualifying.</p>
              </article>
              <figure className="dashboard-widget widget-image widget-strategy"><img src={PHOTO.pit} alt="Aston Martin pit crew working on an F1 car" loading="lazy"/><figcaption><span>RACE OPERATIONS</span><b>Strategy begins before the pit window.</b><PhotoCredit photo="pit"/></figcaption></figure>
              <figure className="dashboard-widget widget-image widget-circuit"><img src={PHOTO.circuit} alt="Interlagos circuit aerial" loading="lazy"/><figcaption><span>CIRCUIT CONTEXT</span><b>Track layout changes the value of pace.</b><PhotoCredit photo="circuit"/></figcaption></figure>
              <article className="dashboard-widget widget-archive">
                <div className="widget-heading"><span>THE LOCAL ARCHIVE</span><b>Evidence available to the predictors</b></div>
                <div className="archive-rows"><div><span>Race and sprint results</span><b>{(overview?.race_result_rows || 0).toLocaleString()}</b></div><div><span>Qualifying classifications</span><b>{(overview?.qualifying_rows || 0).toLocaleString()}</b></div><div><span>Practice classifications</span><b>{(overview?.practice_rows || 0).toLocaleString()}</b></div><div><span>Lap observations</span><b>{(overview?.session_lap_rows || 0).toLocaleString()}</b></div></div>
                <small>Counts reflect local tables currently loaded by the API; availability varies by event and session.</small>
              </article>
            </section>

            <section className="upcoming-section">
              <div className="section-heading-row"><div><span className="section-number">CALENDAR / UPCOMING</span><h3>Next rounds on the calendar</h3><p>Select a Grand Prix to use it in the prediction and practice tools.</p></div><span className="calendar-range">{events.length} upcoming entries</span></div>
              {events.length ? <div className="event-grid">{events.slice(0,8).map(event => <button key={`${event.season}-${event.round}`} className={`event-card ${selected?.season === event.season && selected?.round === event.round ? 'selected' : ''}`} onClick={() => chooseEvent(event)}><span className="event-card-country"><CountryFlag country={event.country}/><small>{event.country || 'Country unavailable'}</small><i>ROUND {String(event.round).padStart(2,'0')}</i></span><b>{event.race_name}</b><span>{event.circuit_name || event.locality || 'Circuit information unavailable'}</span><div className="event-card-bottom"><small>{dateFmt(event.start_utc)}</small><ChevronRight size={15}/></div></button>)}</div> : <Empty title="No upcoming event found" body="Check that the local calendar contains future rounds and verified race start times."/>}
            </section>
            <div className="method-note"><AlertTriangle size={17}/><p><b>Read every forecast with its coverage note.</b> Current-session data, long-run samples, weather and historical circuit records do not exist for every Grand Prix. Missing inputs are reported rather than treated as measured zeroes.</p></div>
          </>}

          {active === 'Pole Predictor' && <>
            <PageHeader kicker="QUALIFYING / BEFORE Q1" title="Who has the one-lap pace?" detail="The pole forecast combines a trained classifier with the agreed evidence hierarchy: current-session pace, constructor strength, driver form, teammate pace, weather and circuit history." right={<EventSelect events={events} selected={selected} onChange={chooseEvent}/>}/>
            <PredictionPanel kind="pole" selected={selected} detail={eventDetail} prediction={prediction} error={predictionError} loading={loadingPrediction} onRun={runPrediction}
              poleForecastStage={poleForecastStage} onPoleForecastStageChange={stage => { setPoleForecastStage(stage); setPrediction(null); setPredictionError(''); }}
              poleCutoff={poleCutoff} onPoleCutoffChange={cutoff => { setPoleCutoff(cutoff); setPrediction(null); setPredictionError(''); }}/>
          </>}
          {active === 'Race Predictor' && <>
            <PageHeader kicker="GRAND PRIX / BEFORE LIGHTS OUT" title="Who wins when pace meets track position?" detail="Choose whether the model should use the Pole Predictor's projected grid before qualifying or the official starting grid after qualifying." right={<EventSelect events={events} selected={selected} onChange={chooseEvent}/>}/>
            <PredictionPanel kind="race" selected={selected} detail={eventDetail} prediction={prediction} error={predictionError} loading={loadingPrediction} onRun={runPrediction} raceStage={raceStage} onRaceStageChange={(stage) => { setRaceStage(stage); setPrediction(null); setPredictionError(''); }}
              poleForecastStage={poleForecastStage} onPoleForecastStageChange={stage => { setPoleForecastStage(stage); setPrediction(null); setPredictionError(''); }}
              poleCutoff={poleCutoff} onPoleCutoffChange={cutoff => { setPoleCutoff(cutoff); setPrediction(null); setPredictionError(''); }}/>
          </>}
          {active === 'Practice Intelligence' && <>
            <PageHeader kicker="WEEKEND INTELLIGENCE / SESSION DATA" title="Practice tells us how the car behaves." detail="Browse every loaded Grand Prix weekend in the 2018–2026 archive. Official Jolpica/FastF1 classifications are preferred; fastest-lap tables are reconstructed from FastF1/OpenF1 lap archives only when an official session classification is absent." right={<EventSelect events={archiveEvents} selected={practiceSelected} onChange={setPracticeSelected} label="GRAND PRIX ARCHIVE" placeholder="Select a Grand Prix"/>}/>
            <section className="panel"><div className="panel-heading"><div><div className="panel-kicker">HISTORICAL PRACTICE / FP1 · FP2 · FP3</div><h3>{titleEvent(practiceSelected)}</h3><p>{practiceRows.length ? `${practiceRows.length} driver-session records available.` : (practiceData?.detail || 'Loading this Grand Prix practice archive.')}</p></div><StatusPill tone={practiceRows.length ? 'success' : 'warning'}>{practiceRows.length ? 'PRACTICE DATA AVAILABLE' : 'DATA NOT FOUND'}</StatusPill></div>
              <div className="practice-source-note">Sources: {practiceData?.row_sources?.join(' · ') || 'Awaiting source records'}. Lap-derived rows are fastest valid laps from available lap archives and are labelled separately from official classification results.</div>
            </section>
            {['FP1','FP2','FP3'].map(sessionCode => {
              const sessionRows = practiceRows.filter(row => row.session === sessionCode);
              const sessionInfo = practiceData?.sessions?.find(item => item.session === sessionCode);
              return <section className="panel" key={sessionCode}>
                <div className="panel-heading"><div><div className="panel-kicker">FREE PRACTICE SESSION</div><h3>{sessionCode}</h3><p>{sessionInfo?.rows ? `${sessionInfo.rows} drivers classified.` : 'No records found for this session in the local archive.'}</p></div><StatusPill tone={sessionRows.length ? 'success' : 'warning'}>{sessionRows.length ? 'AVAILABLE' : 'MISSING'}</StatusPill></div>
                {sessionRows.length ? <DataTable rows={sessionRows} columns={[
                  {key:'position',label:'Position'}, {key:'driver_name',label:'Driver'},
                  {key:'constructor_name',label:'Constructor'}, {key:'time',label:'Fastest lap'},
                  {key:'laps',label:'Laps'}, {key:'source',label:'Data source'}
                ]}/> : <Empty title={`${sessionCode} results unavailable`} body="This session may not have taken place in this sprint-format weekend, or its data has not yet been collected. Missing sessions are not inferred."/>}
              </section>;
            })}
          </>}
          {active === 'Constructor Intelligence' && <>
            <PageHeader kicker="CAR PERFORMANCE / TEAM PACE" title="The car sets the performance envelope." detail="Browse current and previous Grand Prix weekends from the 2018–2026 archive. Six evidence-backed components separate one-lap pace, race pace, long-run behaviour, tyre trends, reliability and circuit fit." right={<EventSelect events={archiveEvents} selected={constructorSelected} onChange={chooseConstructorEvent} label="GRAND PRIX ARCHIVE" placeholder="Select a Grand Prix"/>}/>
            {constructorIntelError && <div className="error-banner"><AlertTriangle size={17}/><span>{constructorIntelError}</span></div>}
            {!constructorIntel && !constructorIntelError && <div className="panel"><div className="loading-state"><div className="loader"/><b>Loading constructor intelligence</b><span>Checking the selected event's current and historical data coverage.</span></div></div>}
            {constructorIntel && <>
              <div className="panel-heading"><div><div className="panel-kicker">{titleEvent(constructorSelected)} / FP3 CUTOFF</div><h3>Constructor intelligence coverage</h3><p>Coverage counts show how many constructors have at least one observation in each component.</p></div><StatusPill tone={constructorIntel.available ? 'success' : 'warning'}>{constructorIntel.available ? 'FEATURES AVAILABLE' : 'DATA INCOMPLETE'}</StatusPill></div>
              <section className="component-grid">
                {[
                  ['Qualifying pace','Normalised one-lap and soft-tyre pace gaps','qualifying_pace'],
                  ['Race pace','Historical race pace and current weekend long-run pace','race_pace'],
                  ['Long-run pace','Valid long-run laps, stints and historical race-lap coverage','long_run_pace'],
                  ['Tyre behaviour','Compound-specific lap-time slope versus tyre age','tyre_behaviour'],
                  ['Reliability','Recent DNF and mechanical-retirement history','reliability'],
                  ['Circuit suitability','Same-circuit results, grid conversion and SC/VSC history','circuit_suitability'],
                ].map(([label, description, key]) => {
                  const count = constructorIntel.pillar_coverage?.[key] || 0;
                  const ready = count > 0;
                  return <div className="component-card" key={key}><Gauge size={18}/><b>{label}</b><p>{description}</p><small>{count} / {constructorIntel.constructor_count} constructors with data</small><StatusPill tone={ready ? 'success' : 'warning'}>{ready ? 'DATA AVAILABLE' : 'AWAITING DATA'}</StatusPill></div>;
                })}
              </section>
              <section className="panel"><div className="panel-heading"><div><div className="panel-kicker">TEAM COMPARISON / EXPLAINED METRICS</div><h3>{titleEvent(constructorSelected)}</h3><p>Only metrics with real observations for at least one constructor are shown. Values are labelled with their units and interpreted below.</p></div></div>
                {constructorIntel.constructors?.length ? <>
                  <DataTable rows={constructorIntel.constructors} columns={availableConstructorMetricColumns(constructorIntel.constructors)}/>
                  <div className="metric-guide-intro"><b>How to read this table</b><span>For pace gaps and tyre-age trends, values closer to zero are generally better. A negative tyre-age trend can mean the lap-time sample improved as the tyre aged, which is usually caused by track evolution or sample noise rather than tyres getting faster. Lower average finish, pit duration and DNF rate are generally preferable. Percentages are historical frequencies, not predictions.</span></div><div className="metric-guide-grid">{availableConstructorMetricColumns(constructorIntel.constructors).filter(column => column.key !== 'constructor_name').map(column => <div className="metric-guide-item" key={column.key}><b>{column.label}</b><p>{column.description}</p></div>)}</div>
                </> : <Empty title="No constructor rows" body="A current event roster is not available yet."/>}
              </section>
              <p className="footnote">{constructorIntel.note}</p>
            </>}
          </>}
          {active === 'Driver Intelligence' && <>
            <PageHeader kicker="DRIVER PERFORMANCE / RELATIVE CONTRIBUTION" title="Separate the driver from the car." detail="The target is teammate-relative driver contribution, not raw finishing position alone." right={<select className="select-compact" value={season || ''} onChange={e => setSeason(Number(e.target.value))}>{(overview?.seasons || []).slice().reverse().map(s => <option key={s} value={s}>{s}</option>)}</select>}/>
            <StandingsPanel data={standings?.drivers || []} kind="driver" note={standings?.source}/>
          </>}
          {active === 'Circuit Intelligence' && <>
            <PageHeader kicker="CIRCUIT PROFILE / HISTORICAL CONTEXT" title="Every circuit asks a different question." detail={`All distinct F1 circuits present in the ${overview?.season_range || '2018–2026'} calendar, with historical Grand Prix records.`}/>
            <section className="panel"><div className="panel-heading"><div><div className="panel-kicker">CIRCUIT DIRECTORY / {overview?.season_range || '2018–2026'}</div><h3>{circuits.length} circuits used since 2018</h3><p>Each circuit opens every available Grand Prix at that venue, including alternate event titles held at the same track.</p></div></div><div className="circuit-grid">{circuits.map(c => <button className="circuit-card" key={c.circuit_id} onClick={() => { setHistory(null); request(`/api/history/${encodeURIComponent(c.circuit_id)}?season=2018&last_season=${Math.max(...(overview?.seasons || [2026]))}`).then(setHistory).catch(err => setPredictionError(err.message)); }}><span className="circuit-country"><CountryFlag country={c.country}/>{c.country || 'Country unavailable'} · {c.first_used_season}–{c.last_used_season}</span><b>{c.circuit_name || c.circuit_id}</b><small>{(c.grand_prix_names || []).join(' · ') || c.locality || 'Grand Prix circuit'}</small><em>{c.event_count} calendar entries</em></button>)}</div>{history && <div className="history-result"><h4>{history.circuit_name || history.circuit_id} · {history.season_range} history</h4>{history.rows?.length ? <DataTable rows={history.rows} columns={Object.keys(history.rows[0]).slice(0,10)}/> : <Empty title="No circuit history available" body="The loaded race results do not contain completed records for this circuit in the selected year range."/>}</div>}</section>
          </>}
          {active === 'Season Standings' && <>
            <PageHeader kicker="CHAMPIONSHIP / SEASON TO DATE" title="The championship as it stood." detail="Standings are calculated from locally collected race and sprint result rows through the selected season." right={<select className="select-compact" value={season || ''} onChange={e => setSeason(Number(e.target.value))}>{(overview?.seasons || []).slice().reverse().map(s => <option key={s} value={s}>{s}</option>)}</select>}/>
            <div className="standings-grid"><StandingsPanel data={standings?.drivers || []} kind="driver"/><StandingsPanel data={standings?.constructors || []} kind="constructor"/></div><p className="footnote">Local fallback calculations may differ from official FIA/F1 tie-break rules and do not independently verify penalties or corrections not present in the source data.</p>
          </>}
          {active === 'Model Lab' && <>
            <PageHeader kicker="MODEL VALIDATION / TRANSPARENCY" title="Measure the model honestly." detail="Chronological holdouts, event-level ranking performance and probability calibration matter more than a single headline accuracy."/>
            <div className="method-warning"><AlertTriangle size={18}/><div><b>Hybrid models — experimental until backtested</b><p>Current inference combines the trained models with transparent expert-priority scores. The new constructor, long-run, tyre, pit-stop and reliability inputs must be retrained and evaluated on chronological unseen Grand Prix weekends before any accuracy claim is made.</p></div></div>
            <section className="panel"><div className="panel-heading"><div><div className="panel-kicker">LOCAL EVALUATION FILE</div><h3>Training and test metrics</h3></div><StatusPill tone={metrics?.available ? 'success' : 'warning'}>{metrics?.available ? 'METRICS FOUND' : 'NOT FOUND'}</StatusPill></div>{metrics?.available ? <pre className="metrics-json">{JSON.stringify(metrics.metrics, null, 2)}</pre> : <Empty title="No local metrics file found" body={metrics?.detail || 'Copy the locally generated metrics.json into artifacts/ to inspect the previous model evaluation.'}/>}</section>
          </>}
        </>}
        <footer className="workspace-footer"><span>THE GRAND PRIX PREDICTOR</span><span>Independent motorsport analysis · Not affiliated with Formula 1 or the FIA</span></footer>
      </div>
    </main>
  </div>;
}

function EventSelect({ events, selected, onChange, label = 'EVENT', placeholder = 'Select upcoming GP' }) {
  return <label className="event-select"><span>{label}</span><select value={selected ? `${selected.season}-${selected.round}` : ''} onChange={e => { const event = events.find(x => `${x.season}-${x.round}` === e.target.value); if (event) onChange(event); }}><option value="" disabled>{placeholder}</option>{events.map(e => <option key={`${e.season}-${e.round}`} value={`${e.season}-${e.round}`}>{e.season} · R{String(e.round).padStart(2,'0')} · {e.race_name}</option>)}</select></label>;
}
function PredictionPanel({ kind, selected, detail, prediction, error, loading, onRun, raceStage = 'after_qualifying', onRaceStageChange,
  poleForecastStage = 'after_free_practice', onPoleForecastStageChange, poleCutoff = 'FP3', onPoleCutoffChange }) {
  const rows = prediction?.predictions || [];
  const availableSessions = new Set((detail?.practice_sessions || []).filter(item => item.available).map(item => item.session));
  const requiredByCutoff = { FP1: ['FP1'], FP2: ['FP1', 'FP2'], FP3: ['FP1', 'FP2', 'FP3'] };
  const missingCutoffSessions = (requiredByCutoff[poleCutoff] || []).filter(code => !availableSessions.has(code));
  const hasSelectedPractice = missingCutoffSessions.length === 0;
  const futurePoleWindow = Boolean(selected?.pole_prediction_available);
  const poleStageAvailable = poleForecastStage === 'before_free_practice' || hasSelectedPractice;
  const projectedGridAvailable = poleStageAvailable;
  const available = kind === 'pole' ? futurePoleWindow && poleStageAvailable : raceStage === 'before_qualifying' ? Boolean(selected?.race_prediction_before_qualifying_available && projectedGridAvailable) : selected?.race_prediction_available;
  const resultsKnown = kind === 'pole' ? selected?.has_qualifying_result : selected?.has_race_result;
  return <div className="prediction-layout">
    <section className="panel prediction-controls"><div className="panel-kicker">{kind === 'pole' ? 'PRE-QUALIFYING FORECAST' : raceStage === 'before_qualifying' ? 'BEFORE QUALIFYING · POLE PREDICTOR GRID' : 'AFTER QUALIFYING · OFFICIAL GRID'}</div><h3>{titleEvent(selected)}</h3><p>{selected?.circuit_name || 'Circuit information unavailable'} · {selected?.country || 'Location unavailable'}</p>{kind === 'pole' && <><div className="race-stage-switch pole-stage-switch" role="group" aria-label="Pole forecast stage"><button type="button" className={poleForecastStage === 'before_free_practice' ? 'active' : ''} onClick={() => onPoleForecastStageChange?.('before_free_practice')}>Before Free Practice</button><button type="button" className={poleForecastStage === 'after_free_practice' ? 'active' : ''} onClick={() => onPoleForecastStageChange?.('after_free_practice')}>After Free Practice</button></div>{poleForecastStage === 'after_free_practice' && <div className="race-stage-switch cutoff-switch" role="group" aria-label="Practice cutoff">{['FP1','FP2','FP3'].map(code => <button type="button" key={code} disabled={!requiredByCutoff[code].every(required => availableSessions.has(required))} className={poleCutoff === code ? 'active' : ''} onClick={() => onPoleCutoffChange?.(code)}>{code}</button>)}</div>}</>}{kind === 'race' && <div className="race-stage-switch" role="group" aria-label="Race prediction stage"><button type="button" className={raceStage === 'before_qualifying' ? 'active' : ''} onClick={() => onRaceStageChange?.('before_qualifying')}>Before Qualifying</button><button type="button" className={raceStage === 'after_qualifying' ? 'active' : ''} onClick={() => onRaceStageChange?.('after_qualifying')}>After Qualifying</button></div>}{kind === 'race' && raceStage === 'before_qualifying' && <><p className="hint">Projected grid source: Pole Predictor · {poleForecastStage === 'before_free_practice' ? 'Before Free Practice' : `After ${poleCutoff}`}</p><div className="race-stage-switch pole-stage-switch" role="group" aria-label="Projected grid forecast stage"><button type="button" className={poleForecastStage === 'before_free_practice' ? 'active' : ''} onClick={() => onPoleForecastStageChange?.('before_free_practice')}>Before FP</button><button type="button" className={poleForecastStage === 'after_free_practice' ? 'active' : ''} onClick={() => onPoleForecastStageChange?.('after_free_practice')}>After FP</button></div>{poleForecastStage === 'after_free_practice' && <div className="race-stage-switch cutoff-switch" role="group" aria-label="Projected grid practice cutoff">{['FP1','FP2','FP3'].map(code => <button type="button" key={code} disabled={!requiredByCutoff[code].every(required => availableSessions.has(required))} className={poleCutoff === code ? 'active' : ''} onClick={() => onPoleCutoffChange?.(code)}>{code}</button>)}</div>}</>}<div className="session-state"><div><small>QUALIFYING DATA</small><b>{selected?.has_qualifying_result ? 'PUBLISHED' : 'NOT PUBLISHED IN LOCAL DATA'}</b></div><div><small>RACE RESULT</small><b>{selected?.has_race_result ? 'COMPLETED' : 'NOT COMPLETED'}</b></div><div><small>PRACTICE ROWS</small><b>{detail?.practice?.length ?? 0}</b></div></div>
      {resultsKnown && <div className="inline-note warning"><AlertTriangle size={16}/>{kind === 'pole' ? 'Qualifying has already been recorded. This event is closed to pole forecasting.' : 'The race is complete. Use it for backtesting, not live prediction.'}</div>}
      <button className="button-dark full" onClick={() => onRun(kind)} disabled={!selected || loading || !available}>{loading ? 'Calculating…' : kind === 'pole' ? (poleForecastStage === 'before_free_practice' ? 'Run pre-practice pole forecast' : `Run ${poleCutoff} pole forecast`) : raceStage === 'before_qualifying' ? 'Predict race using Pole Predictor grid' : 'Predict race using official grid'}<ArrowRight size={16}/></button>
      {!available && !resultsKnown && <p className="hint">{kind === 'pole' ? (poleForecastStage === 'before_free_practice' ? 'Uses historical-only form and constructor strength before FP1.' : `The ${poleCutoff} forecast requires ${requiredByCutoff[poleCutoff].join(' + ')}. Missing locally: ${missingCutoffSessions.join(', ') || 'model not ready'}. Choose the latest enabled cutoff.`) : raceStage === 'before_qualifying' ? `The projected grid requires the selected Pole Predictor stage. Missing practice sessions: ${missingCutoffSessions.join(', ') || 'none; check model artifact readiness'}.`  : 'This mode requires an upcoming event with qualifying results and a race start still in the future.'}</p>}
      <div className="inline-note"><AlertTriangle size={16}/><span>Forecast quality depends on practice, long-run, grid, weather and historical coverage. Missing inputs are reported rather than treated as measured zeroes.</span></div>
    </section>
    <section className="panel results-panel"><div className="panel-heading"><div><div className="panel-kicker">MODEL OUTPUT</div><h3>{kind === 'pole' ? 'Pole contender ranking' : 'Race outcome ranking'}</h3></div><span className="model-state">{prediction ? "Forecast returned · experimental" : "No forecast generated"}</span></div>
      {error && <div className="inline-note warning"><AlertTriangle size={16}/>{error}</div>}
      {!rows.length ? <Empty title="No prediction generated" body="Choose an eligible upcoming Grand Prix and run the forecast. The API will return a data-coverage warning when required inputs are missing."/> : <><div className="prediction-warning">{prediction.starting_grid_source === 'pole_predictor_predicted_grid' ? `Race model uses the projected grid generated by the Pole Predictor at ${prediction.practice_cutoff || 'the selected pre-qualifying stage'}. ` : ''}{(prediction.warnings || []).join(' ') } {prediction.probability_note}</div>{kind === 'race' && prediction.pole_predictor_grid?.length > 0 && <div className="predicted-grid-preview"><b>Projected starting grid · Pole Predictor ({prediction.practice_cutoff})</b><DataTable rows={prediction.pole_predictor_grid} columns={['predicted_grid_position','driver_name','constructor_name','pole_score_share']}/></div>}<div className="prediction-table">{kind === 'pole' ? <><div className="prediction-row heading"><span>#</span><span>DRIVER</span><span>CONSTRUCTOR</span><span>RELATIVE SCORE</span><span>MODEL SCORE</span></div>{rows.map((r,i) => <div className="prediction-row" key={`${r.driver_name}-${i}`}><b>{i+1}</b><strong className="table-entity-name"><CountryFlag code={driverFlagFor(r)}/>{r.driver_name}</strong><span>{r.constructor_name}</span><span>{pct(r.pole_score_share)}</span><span>{pct(r.pole_probability)}</span></div>)}</> : <><div className="prediction-row race heading"><span>#</span><span>DRIVER</span><span>CONSTRUCTOR</span><span>START GRID</span><span>EXPECTED POS.</span><span>WIN</span><span>PODIUM</span><span>DNF</span></div>{rows.map((r,i) => <div className="prediction-row race" key={`${r.driver_name}-${i}`}><b>{r.predicted_order || i+1}</b><strong className="table-entity-name"><CountryFlag code={driverFlagFor(r)}/>{r.driver_name}</strong><span>{r.constructor_name}</span><span>{fmt(r.starting_grid_position,0)}</span><span>{fmt(r.expected_finish_position)}</span><span>{pct(r.win_probability)}</span><span>{pct(r.podium_probability)}</span><span>{pct(r.dnf_probability)}</span></div>)}</>}</div></>}
    </section>
  </div>;
}
function StandingsPanel({ data, kind, note }) {
  return <section className="panel standings-panel"><div className="panel-heading"><div><div className="panel-kicker">{kind === 'driver' ? 'DRIVERS' : 'CONSTRUCTORS'}</div><h3>{kind === 'driver' ? 'Driver championship' : 'Constructor championship'}</h3></div></div>{data?.length ? <div className="standings-table">{data.slice(0,20).map((row,i) => {
    const flag = kind === 'driver' ? driverFlagFor(row) : constructorFlagFor(row);
    return <div className="standing-row" key={row.driver_id || row.constructor_id || i}><i>{row.position || i+1}</i><b className="standing-name"><CountryFlag code={flag}/>{row.driver || row.constructor}</b><span>{row.constructor || ''}</span><strong>{fmt(row.points,0)} <small>pts</small></strong></div>;
  })}</div> : <Empty title="Standings unavailable" body="No results for this season are available in the local dataset."/>}{note && <p className="footnote">{note}</p>}</section>;
}

function DataTable({ rows, columns }) {
  if (!rows?.length) return <Empty title="No rows" body="No records are available."/>;
  const defs = columns.map(column => typeof column === 'string' ? { key: column, label: column.replaceAll('_',' ') } : column);
  return <div className="data-table-wrap"><table className="data-table"><thead><tr>{defs.map(column => <th key={column.key} title={column.description || ''}>{column.label}</th>)}</tr></thead><tbody>{rows.map((r,i) => <tr key={i}>{defs.map(column => {
    const value = r[column.key];
    let display = value === null || value === undefined || value === '' ? '—' : String(value);
    if (value !== null && value !== undefined && column.format === 'percent') display = `${(Number(value) * 100).toFixed(1)}%`;
    else if (value !== null && value !== undefined && column.format === 'gap') display = `${Number(value) > 0 ? '+' : ''}${Number(value).toFixed(3)}%`;
    else if (value !== null && value !== undefined && column.format === 'seconds') display = `${Number(value).toFixed(2)} s`;
    else if (value !== null && value !== undefined && column.format === 'seconds-per-lap') display = `${Number(value) > 0 ? '+' : ''}${Number(value).toFixed(3)} s/lap`;
    else if (value !== null && value !== undefined && column.format === 'position') display = `P${Number(value).toFixed(1)}`;
    else if (value !== null && value !== undefined && column.format === 'number') display = Number(value).toFixed(1);
    const flag = column.key === 'driver_name' || column.key === 'driver' ? driverFlagFor(r) : column.key === 'constructor_name' || column.key === 'constructor' ? constructorFlagFor(r) : null;
    return <td key={column.key}>{flag ? <span className="table-entity-name"><CountryFlag code={flag}/>{display}</span> : display}</td>;
  })}</tr>)}</tbody></table></div>;
}

const CONSTRUCTOR_METRIC_COLUMNS = [
  { key:'weekend_best_one_lap_gap_pct', label:'Best practice lap gap', format:'gap', description:'How far the constructor’s fastest practice lap was behind the fastest car in percentage terms. Closer to 0% is faster.' },
  { key:'latest_session_one_lap_gap_pct', label:'Latest session pace gap', format:'gap', description:'The constructor’s best-lap gap in the latest completed practice session. This shows recent changes in the order.' },
  { key:'historical_qualifying_gap_pct', label:'Recent qualifying pace gap', format:'gap', description:'Average qualifying pace gap over recent Grands Prix. Lower is faster.' },
  { key:'season_qualifying_gap_pct', label:'Season qualifying pace gap', format:'gap', description:'Constructor’s average qualifying pace gap in the current season up to this event.' },
  { key:'current_weekend_longrun_gap_pct', label:'Current weekend long-run gap', format:'gap', description:'Representative long-run pace relative to the fastest constructor with sufficient stint data.' },
  { key:'historical_race_longrun_gap_pct', label:'Recent race long-run gap', format:'gap', description:'Historical race-stint pace gap based on previous events where lap data are available.' },
  { key:'overall_tyre_age_slope_sec_per_lap', label:'Tyre-age trend', format:'seconds-per-lap', description:'Observed lap-time change per additional tyre-age lap. This is a proxy, not a pure tyre-degradation measurement.' },
  { key:'soft_tyre_age_slope_sec_per_lap', label:'Soft tyre-age trend', format:'seconds-per-lap', description:'Observed lap-time change per tyre-age lap on soft tyres.' },
  { key:'medium_tyre_age_slope_sec_per_lap', label:'Medium tyre-age trend', format:'seconds-per-lap', description:'Observed lap-time change per tyre-age lap on medium tyres.' },
  { key:'hard_tyre_age_slope_sec_per_lap', label:'Hard tyre-age trend', format:'seconds-per-lap', description:'Observed lap-time change per tyre-age lap on hard tyres.' },
  { key:'recent_mean_pit_duration_sec', label:'Average pit-stop duration', format:'seconds', description:'Average recorded pit-stop duration from recent available events. It is not identical to stationary tyre-change time.' },
  { key:'recent_five_gp_dnf_rate', label:'Recent DNF rate', format:'percent', description:'Share of the constructor’s recent Grand Prix starts that ended in retirement/non-classification.' },
  { key:'recent_five_gp_mechanical_dnf_rate', label:'Mechanical DNF rate', format:'percent', description:'Share of recent starts marked as mechanical/non-driver-fault retirements where status labels allow classification.' },
  { key:'same_circuit_average_finish', label:'Average finish at this circuit', format:'position', description:'Average finishing position on previous visits to this circuit. Lower is better.' },
  { key:'same_circuit_average_positions_gained', label:'Average positions gained', format:'number', description:'Average grid positions gained or lost at this circuit compared with the starting position. Positive means gained.' },
  { key:'same_circuit_grid_win_conversion', label:'Grid-to-win conversion', format:'percent', description:'Historical share of starts from the relevant grid group that converted into a win at this circuit.' },
  { key:'historical_safety_car_event_rate', label:'Safety Car frequency', format:'percent', description:'Share of prior events at this circuit with a Safety Car message in the available race-control archive.' },
  { key:'historical_virtual_safety_car_event_rate', label:'Virtual Safety Car frequency', format:'percent', description:'Share of prior events at this circuit with a VSC message in the available race-control archive.' },
];
function availableConstructorMetricColumns(rows) {
  const visible = CONSTRUCTOR_METRIC_COLUMNS.filter(column => rows.some(row => row[column.key] !== null && row[column.key] !== undefined && String(row[column.key]).trim() !== '' && Number.isFinite(Number(row[column.key]))));
  return [{ key:'constructor_name', label:'Constructor', description:'Formula One constructor/team.' }, ...visible];
}

createRoot(document.getElementById('root')).render(<App/>);
