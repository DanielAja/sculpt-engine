//! Integration tests for the pure-Rust document, edit, export/import, and voice bridge.

use sculpt_engine::bridge::{self, commands, edit, io, voice};
use serde_json::Value;

#[test]
fn starter_doc_commands_undo_and_export() {
    let reg = bridge::BridgeRegistry::global();
    let h = reg.new_doc();

    let summary: Value = serde_json::from_str(&bridge::with_doc(h, String::new(), |d| d.summary_json())).unwrap();
    assert_eq!(summary["objects"], 1);
    assert!(summary["triangles"].as_u64().unwrap() > 0);

    // Add a sphere
    let add_res: Value = serde_json::from_str(&bridge::with_doc(h, String::new(), |d| {
        commands::doc_apply(d, r#"{"addPrimitive":{"_0":{"kind":"sphere","size":[20,20,20],"segments":16,"rings":8},"position":[25,0,10],"name":"Ball"}}"#)
    }))
    .unwrap();
    assert_eq!(add_res["ok"], true);
    let created = add_res["created"].as_array().unwrap();
    assert_eq!(created.len(), 1);
    let ball_id = created[0].as_str().unwrap().to_string();

    // State shows 2 objects and undo available
    let state: Value = serde_json::from_str(&bridge::with_doc(h, String::new(), |d| d.state_json())).unwrap();
    assert_eq!(state["objects"].as_array().unwrap().len(), 2);
    assert_eq!(state["canUndo"], true);

    // Render mesh + pick + edit mode + extrude
    let rm = bridge::with_doc(h, None, |d| d.render_mesh(&ball_id)).unwrap();
    assert!(!rm.positions.is_empty());
    assert!(!rm.indices.is_empty());

    let em: Value = serde_json::from_str(&bridge::with_doc(h, String::new(), |d| edit::set_edit_mode(d, "face"))).unwrap();
    assert_eq!(em["ok"], true);
    let ov = bridge::with_doc(h, None, |d| edit::build_edit_overlay(d, 0.5, true)).unwrap();
    assert!(!ov.line_positions.is_empty());

    // Project save + round-trip reencode
    let saved = bridge::with_doc(h, Vec::new(), |d| d.to_project_file(Some(r#"{"yaw":0.5}"#)).encode().into_bytes());
    let reencoded = sculpt_engine::format::ProjectFile::decode(&saved).unwrap().encode().into_bytes();
    assert_eq!(saved, reencoded);

    // Snapshot exports: STL, OBJ, 3MF, GLB, USDZ + PrintCheck
    let snap = bridge::with_doc(h, None, |d| Some(bridge::SceneSnapshot { scene: d.scene.clone(), selection: d.selection.clone(), revision: d.revision })).unwrap();
    let stl = io::export_stl(&snap, None).unwrap();
    assert!(stl.len() > 84);
    for fmt in ["stl", "obj", "3mf", "glb", "usdz"] {
        let files = io::export_files(&snap, &format!(r#"{{"format":"{fmt}","fileName":"TestModel"}}"#)).unwrap();
        assert!(!files.is_empty());
        assert!(!files[0].data.is_empty());
    }
    let pc: Value = serde_json::from_str(&io::print_check(&snap, "{}")).unwrap();
    assert_eq!(pc["printable"], true);

    // Undo brings back to 1 object
    assert!(bridge::with_doc(h, false, |d| d.undo()));
    let state2: Value = serde_json::from_str(&bridge::with_doc(h, String::new(), |d| d.state_json())).unwrap();
    assert_eq!(state2["objects"].as_array().unwrap().len(), 1);

    reg.free_doc(h);
}

#[test]
fn voice_quick_plan_and_cloud_providers() {
    let mug: Value = serde_json::from_str(&voice::call("quickPlan", r#"{"request":"a red mug","sceneNames":[]}"#, None)).unwrap();
    assert_eq!(mug["ok"], true);
    let steps = mug["plan"]["steps"].as_array().unwrap();
    assert!(steps.iter().any(|s| s["action"] == "boolean" && s["booleanOp"] == "difference"));

    let provs: Value = serde_json::from_str(&voice::call("providers", "{}", None)).unwrap();
    assert_eq!(provs["ok"], true);
    assert_eq!(provs["consentVersion"], 1);
    assert_eq!(provs["providers"].as_array().unwrap().len(), 6);

    let pretty: Value = serde_json::from_str(&voice::call("prettyModel", r#"{"id":"claude-opus-5-5"}"#, None)).unwrap();
    assert_eq!(pretty["name"], "Claude Opus 5.5");
}
