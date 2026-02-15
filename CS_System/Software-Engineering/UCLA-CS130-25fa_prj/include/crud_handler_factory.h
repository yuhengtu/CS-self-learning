#ifndef CRUD_HANDLER_FACTORY_H
#define CRUD_HANDLER_FACTORY_H

#include <memory>
#include <string>

#include "request_handler_factory.h"
#include "server_config.h"
#include "crud_manager.h"

class CrudHandlerFactory : public RequestHandlerFactory {
public:
  explicit CrudHandlerFactory(const HandlerSpec& spec);
  std::unique_ptr<RequestHandler> create(const std::string& location,
                                         const std::string& url) override;
private:
  std::string data_path_;
  std::shared_ptr<CrudManagerInterface> manager_;
};

void RegisterCrudHandlerFactory();

#endif // CRUD_HANDLER_FACTORY_H
