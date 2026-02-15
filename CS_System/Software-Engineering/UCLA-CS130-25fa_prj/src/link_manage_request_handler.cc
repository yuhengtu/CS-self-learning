#include "link_manage_request_handler.h"

#include <boost/json.hpp>
#include <openssl/sha.h>
#include <array>
#include <optional>
#include <random>
#include <sstream>
#include <string_view>

#include "link_record_serialization.h"
#include "response_builder.h"

namespace json = boost::json;

namespace {

constexpr std::string_view kPasswordHeader = "Link-Password";

std::string TrimPrefix(const std::string& s, const std::string& prefix) {
  if (s.rfind(prefix, 0) == 0) return s.substr(prefix.size());
  return s;
}

std::string HexEncode(const unsigned char* data, std::size_t len) {
  static constexpr std::string_view kHexDigits = "0123456789abcdef";
  std::string out(len * 2, '\0');
  for (std::size_t i = 0; i < len; ++i) {
    unsigned char byte = data[i];
    out[2 * i] = kHexDigits[(byte >> 4) & 0x0F];
    out[2 * i + 1] = kHexDigits[byte & 0x0F];
  }
  return out;
}

std::string GenerateSalt() {
  std::array<unsigned char, 16> bytes{};
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto& b : bytes) {
    b = static_cast<unsigned char>(dist(gen));
  }
  return HexEncode(bytes.data(), bytes.size());
}

std::string HashPassword(const std::string& password, const std::string& salt) {
  std::string input = salt + password;
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(input.data()), input.size(), digest);
  return HexEncode(digest, SHA256_DIGEST_LENGTH);
}

bool RecordIsProtected(const LinkRecord& rec) {
  return rec.password_hash.has_value() && rec.password_salt.has_value();
}

bool IsAuthorized(const LinkRecord& rec, const std::string& provided_password) {
  if (!RecordIsProtected(rec)) return true;
  if (provided_password.empty()) return false;
  const std::string expected = HashPassword(provided_password, *rec.password_salt);
  return rec.password_hash.has_value() && *rec.password_hash == expected;
}

void BuildForbiddenResponse(response& resp) {
  ResponseBuilder(403, "Forbidden")
      .withBody("missing or invalid password")
      .build(resp);
}

}  // namespace

LinkManageRequestHandler::LinkManageRequestHandler(
    std::string mount_prefix,
    std::shared_ptr<LinkManagerInterface> manager)
    : mount_prefix_(std::move(mount_prefix)), manager_(std::move(manager)) {}

std::unique_ptr<response> LinkManageRequestHandler::handle_request(const request& req) {
  auto out = std::make_unique<response>();
  const std::string rel = TrimPrefix(req.uri, mount_prefix_);
  // rel might be "" or "/{code}"
  if (req.method == "POST" && (rel.empty() || rel == "/")) {
    return HandlePost(req);
  }

  // Extract code if present
  std::string code;
  if (!rel.empty()) {
    if (rel[0] == '/') code = rel.substr(1);
    else code = rel;
  }

  if (code.empty()) {
    ResponseBuilder(405, "Method Not Allowed").withHeader("Allow", "POST, GET, PUT, DELETE").build(*out);
    return out;
  }

  if (req.method == "GET") return HandleGet(code, req);
  if (req.method == "PUT") return HandlePut(code, req);
  if (req.method == "DELETE") return HandleDelete(code, req);

  ResponseBuilder(405, "Method Not Allowed").withHeader("Allow", "POST, GET, PUT, DELETE").build(*out);
  return out;
}

std::unique_ptr<response> LinkManageRequestHandler::HandlePost(const request& req) {
  auto out = std::make_unique<response>();
  // Require JSON
  // Parse minimal JSON {"url":"..."}
  json::error_code ec;
  json::value v = json::parse(req.body, ec);
  if (ec || !v.is_object()) {
    ResponseBuilder::createBadRequest("malformed json").build(*out);
    return out;
  }
  const auto& obj = v.as_object();
  auto* it = obj.if_contains("url");
  if (!it || !it->is_string()) {
    ResponseBuilder::createBadRequest("missing url").build(*out);
    return out;
  }
  std::string url = std::string(it->as_string().c_str());
  // Normalization and validation: handle URL schemes
  if (url.find("://") != std::string::npos) {
    // URL has a scheme - verify it's http or https
    if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0) {
      ResponseBuilder::createBadRequest("only http:// and https:// URLs are supported").build(*out);
      return out;
    }
  } else {
    // No scheme - prepend http://
    url = "http://" + url;
  }
  std::optional<std::string> password;
  if (auto* pw = obj.if_contains("password")) {
    if (!pw->is_string()) {
      ResponseBuilder::createBadRequest("password must be a string").build(*out);
      return out;
    }
    password = std::string(pw->as_string().c_str());
    if (password->empty()) {
      ResponseBuilder::createBadRequest("password cannot be empty").build(*out);
      return out;
    }
  }
  LinkCreateParams p{url};
  if (password) {
    const std::string salt = GenerateSalt();
    p.password_salt = salt;
    p.password_hash = HashPassword(*password, salt);
  }
  auto cr = manager_->Create(p);
  if (cr.status == LinkStatus::Ok && cr.code) {
    std::ostringstream oss;
    oss << "{\"code\":\"" << *cr.code << "\"}";
    ResponseBuilder::createOk().withContentType("application/json").withBody(oss.str()).build(*out);
    return out;
  }
  if (cr.status == LinkStatus::Invalid) {
    ResponseBuilder::createBadRequest("invalid url").build(*out);
    return out;
  }
  ResponseBuilder::createInternalServerError().build(*out);
  return out;
}

std::unique_ptr<response> LinkManageRequestHandler::HandleGet(const std::string& code,
                                                              const request& req) {
  auto out = std::make_unique<response>();
  if (!LinkManagerInterface::IsValidCode(code)) {
    ResponseBuilder::createBadRequest("invalid code").build(*out);
    return out;
  }
  auto gr = manager_->Get(code);
  if (gr.status == LinkStatus::Ok && gr.record) {
    const std::string provided = req.get_header_value(kPasswordHeader);
    if (!IsAuthorized(*gr.record, provided)) {
      BuildForbiddenResponse(*out);
      return out;
    }
    const std::string body = LinkRecordToJsonWithCode(*gr.record);
    ResponseBuilder::createOk().withContentType("application/json").withBody(body).build(*out);
    return out;
  }
  if (gr.status == LinkStatus::NotFound) {
    ResponseBuilder::createNotFound().build(*out);
    return out;
  }
  ResponseBuilder::createInternalServerError().build(*out);
  return out;
}

std::unique_ptr<response> LinkManageRequestHandler::HandlePut(const std::string& code, const request& req) {
  auto out = std::make_unique<response>();
  if (!LinkManagerInterface::IsValidCode(code)) {
    ResponseBuilder::createBadRequest("invalid code").build(*out);
    return out;
  }
  json::error_code ec;
  json::value v = json::parse(req.body, ec);
  if (ec || !v.is_object()) {
    ResponseBuilder::createBadRequest("malformed json").build(*out);
    return out;
  }
  auto* it = v.as_object().if_contains("url");
  if (!it || !it->is_string()) {
    ResponseBuilder::createBadRequest("missing url").build(*out);
    return out;
  }
  std::string url = std::string(it->as_string().c_str());
  // Normalization and validation: handle URL schemes
  if (url.find("://") != std::string::npos) {
    // URL has a scheme - verify it's http or https
    if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0) {
      ResponseBuilder::createBadRequest("only http:// and https:// URLs are supported").build(*out);
      return out;
    }
  } else {
    // No scheme - prepend http://
    url = "http://" + url;
  }
  LinkUpdateParams p{url};
  auto gr = manager_->Get(code);
  if (gr.status == LinkStatus::NotFound) {
    ResponseBuilder::createNotFound().build(*out);
    return out;
  }
  if (gr.status != LinkStatus::Ok || !gr.record) {
    ResponseBuilder::createInternalServerError().build(*out);
    return out;
  }
  const std::string provided = req.get_header_value(kPasswordHeader);
  if (!IsAuthorized(*gr.record, provided)) {
    BuildForbiddenResponse(*out);
    return out;
  }
  auto ur = manager_->Update(code, p);
  if (ur.status == LinkStatus::Ok) {
    std::ostringstream oss;
    oss << "{\"code\":\"" << code << "\"}";
    ResponseBuilder::createOk().withContentType("application/json").withBody(oss.str()).build(*out);
    return out;
  }
  if (ur.status == LinkStatus::Invalid) {
    ResponseBuilder::createBadRequest("invalid url").build(*out);
    return out;
  }
  if (ur.status == LinkStatus::NotFound) {
    ResponseBuilder::createNotFound().build(*out);
    return out;
  }
  ResponseBuilder::createInternalServerError().build(*out);
  return out;
}

std::unique_ptr<response> LinkManageRequestHandler::HandleDelete(const std::string& code,
                                                                 const request& req) {
  auto out = std::make_unique<response>();
  if (!LinkManagerInterface::IsValidCode(code)) {
    ResponseBuilder::createBadRequest("invalid code").build(*out);
    return out;
  }
  auto gr = manager_->Get(code);
  if (gr.status == LinkStatus::FsError) {
    ResponseBuilder::createInternalServerError().build(*out);
    return out;
  }
  if (gr.status == LinkStatus::Invalid) {
    ResponseBuilder::createBadRequest("invalid code").build(*out);
    return out;
  }
  if (gr.status == LinkStatus::Ok && gr.record) {
    const std::string provided = req.get_header_value(kPasswordHeader);
    if (!IsAuthorized(*gr.record, provided)) {
      BuildForbiddenResponse(*out);
      return out;
    }
  }
  auto dr = manager_->Delete(code);
  if (dr.status == LinkStatus::Ok) {
    std::ostringstream oss;
    oss << "{\"code\":\"" << code << "\"}";
    ResponseBuilder::createOk().withContentType("application/json").withBody(oss.str()).build(*out);
    return out;
  }
  ResponseBuilder::createInternalServerError().build(*out);
  return out;
}
    