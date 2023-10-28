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

    let server = HttpServer::new(move || App::new().service(hello).app_data(data_ref.clone()));

    let mut listenfd = ListenFd::from_env();
    let server = match listenfd.take_tcp_listener(0)? {
        Some(listener) => server.listen(listener)?,
        None => server.bind(("127.0.0.1", 3000))?,
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
