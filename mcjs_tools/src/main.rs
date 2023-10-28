use std::time::Duration;

use actix_web::{web, App, HttpResponse, HttpServer, Responder};

use handlebars::Handlebars;
use listenfd::ListenFd;
use serde_json::json;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let mut handlebars = Handlebars::new();
    // TODO Make it independent from the cwd
    handlebars
        .register_templates_directory(".html", "./templates")
        .unwrap();

    let data_ref = web::Data::new(AppData { handlebars });

    let server = HttpServer::new(move || {
        App::new()
            .service(actix_files::Files::new("/assets", "./data/assets/"))
            .service(hello)
            .service(events)
            .app_data(data_ref.clone())
    });

    let mut listenfd = ListenFd::from_env();
    let server = match listenfd.take_tcp_listener(0)? {
        Some(listener) => server.listen(listener)?,
        None => server.bind(("127.0.0.1", 10001))?,
    };

    server.run().await
}

struct AppData<'a> {
    handlebars: Handlebars<'a>,
}

#[actix_web::get("/")]
async fn hello(app_data: web::Data<AppData<'_>>) -> impl Responder {
    let tmpl_params = json! ({
        // some ambition
        "who": "world",
    });

    let body = app_data.handlebars.render("index", &tmpl_params).unwrap();
    HttpResponse::Ok().body(body)
}

#[actix_web::get("/events")]
async fn events(app_data: web::Data<AppData<'static>>) -> impl Responder {
    use actix_web_lab::sse;
    use tokio::sync::mpsc;

    async fn sender_process(app_data: web::Data<AppData<'_>>, tx: mpsc::Sender<sse::Event>) {
        for i in 0..100 {
            let new_body = app_data
                .handlebars
                .render("fragment", &json!({ "event_ndx": i }))
                .unwrap();
            let event = sse::Data::new(new_body).event("content").into();
            tx.send(event).await.expect("could not send event");

            let msecs = rand::random::<u8>() as u64 * 8;
            let dur = Duration::from_millis(msecs);
            tokio::time::sleep(dur).await;
        }
    }

    let (tx, rx) = mpsc::channel(10);
    tokio::spawn(sender_process(app_data.clone(), tx));

    // TODO Figure out *exactly* what this means
    sse::Sse::from_infallible_receiver(rx).with_retry_duration(Duration::from_secs(10))
}
